require 'torch'
require 'nn'
require 'cunn'
require 'nnx'
require 'optim'
require 'image'
require 'pl'

local adversarial = {}

-- training function
function adversarial.train(dataset)
  -- epoch tracker
  epoch = epoch or 1

  -- local vars
  local time = sys.clock()

  -- do one epoch
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')
  for t = 1,dataset:size(),opt.batchSize do

    local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local targets_D = torch.Tensor(opt.batchSize)
    local targets_C = torch.Tensor(opt.batchSize)
    local noise_inputs 
    if type(opt.noiseDim) == 'number' then
      noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)
    else
      noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
    end
    local cond_inputs 
    if type(opt.condDim) == 'number' then
      cond_inputs = torch.Tensor(opt.batchSize, opt.condDim)
    else
      cond_inputs = torch.Tensor(opt.batchSize, opt.condDim[1], opt.condDim[2], opt.condDim[3])
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      --  forward pass
      local outputs = model_D:forward({inputs, cond_inputs})
      local f = criterion_D:forward(outputs[1], targets_D)

      -- backward pass 
      local dD_do = criterion_D:backward(outputs[1], targets_D)
      local dC_do = torch.zeros(outputs[2]:size())
      model_D:backward({inputs, cond_inputs}, {dD_do, dC_do})

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end
      -- update confusion_D (add 1 since classes are binary)
      for i = 1,opt.batchSize do
        local c
        if outputs[1][i][1] > 0.5 then c = 2 else c = 1 end
        confusion_D:add(c, targets_D[i]+1)
      end

      return f,gradParameters_D
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of classifier 
    local fevalC = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients
      cond_inputs:zero() -- make sure cond inputs don't have class info

      --  forward pass
      local outputs = model_D:forward({inputs, cond_inputs})
      local f = criterion_C:forward(outputs[2], targets_C)

      -- backward pass 
      local dD_do =  torch.zeros(outputs[1]:size())
      local dC_do = criterion_C:backward(outputs[2], targets_C) 
      model_D:backward({inputs, cond_inputs}, {dD_do, dC_do})

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end

      -- update confusion_C (add 1 since classes are binary)
      for i = 1,opt.batchSize do
        max, ind = torch.max(outputs[2][i], 1)
        confusion_C:add(ind[1], targets_C[i])
      end

      return f,gradParameters_D
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      if x ~= parameters_G then -- get new parameters
        parameters_G:copy(x)
      end
      
      gradParameters_G:zero() -- reset gradients

      -- forward pass
      local samples = model_G:forward({noise_inputs, cond_inputs})
      local outputs = model_D:forward({samples, cond_inputs})
      local f = criterion_D:forward(outputs[1], targets_D)

      --  backward pass
      local dD_samples = criterion_D:backward(outputs[1], targets_D)
      local dC_samples = torch.zeros(outputs[2]:size())
      model_D:backward({samples, cond_inputs}, {dD_samples, dC_samples})
      local df_do = model_D.gradInput[1]
      model_G:backward({noise_inputs, cond_inputs}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_G:add( sign(parameters_G):mul(opt.coefL1) + parameters_G:clone():mul(opt.coefL2) )
      end

      return f,gradParameters_G
    end

    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    for k=1,opt.K do
      -- (1.1) Real data 
      targets_D:fill(1)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
        -- load new sample
        local sample = dataset[i]
        inputs[k] = sample[1]:clone()
        targets_C[k] = sample[2]
        cond_inputs[k] = sample[3]:clone()
        k = k + 1
      end

      optim.sgd(fevalD, parameters_D, sgdState_D)
      cond_inputs:zero()
      optim.sgd(fevalC, parameters_D, sgdState_D)

      -- (1.2) Sampled data
      noise_inputs:uniform(-1, 1)
      for i=1,opt.batchSize do
        local idx = math.random(dataset:size())
        local sample = dataset[idx]
        cond_inputs[i] = sample[3]:clone()
      end
      local samples = model_G:forward({noise_inputs, cond_inputs})
      inputs:copy(samples)
      targets_D:fill(0)

      optim.sgd(fevalD, parameters_D, sgdState_D)
      cond_inputs:zero()
      optim.sgd(fevalC, parameters_D, sgdState_D)
    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:uniform(-1, 1)
    for i = 1,opt.batchSize do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      cond_inputs[i] = sample[3]:clone()
    end
    targets_D:fill(1)
    optim.sgd(fevalG, parameters_G, sgdState_G)

    -- disp progress
    xlua.progress(t, dataset:size())
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion_D matrix
  print(confusion_D)
  print(confusion_C)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion_D.totalValid * 100}
  confusion_D:zero()
  confusion_C:zero()

  -- save/log current net
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, 'conditional_adversarial.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D = model_D, G = model_G, opt = opt})
  end

  -- next epoch
  epoch = epoch + 1
end

-- test function
function adversarial.test(dataset)
  -- local vars
  local time = sys.clock()

  local targets_D = torch.Tensor(opt.batchSize)
  local targets_C = torch.Tensor(opt.batchSize)
  local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local noise_inputs 
  if type(opt.noiseDim) == 'number' then
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)
  else
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  end
  local cond_inputs 
  if type(opt.condDim) == 'number' then
    cond_inputs = torch.Tensor(opt.batchSize, opt.condDim)
  else
    cond_inputs = torch.Tensor(opt.batchSize, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  end

  print('\n<trainer> on testing Set:')
  for t = 1,dataset:size(),opt.batchSize do
    -- disp progress
    xlua.progress(t, dataset:size())

    ----------------------------------------------------------------------
    -- (1) Real data
    targets_D:fill(1)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      inputs[k] = sample[1]:clone() 
      targets_C[k] = sample[2]
      cond_inputs[k] = sample[3]:clone()
      k = k + 1
    end
    local outputs = model_D:forward({inputs, cond_inputs})
    -- update confusion_D (add 1 since classes are binary)
    for i = 1,opt.batchSize do
      local c
      if outputs[1][i][1] > 0.5 then c = 2 else c = 1 end
      confusion_D:add(c, targets_D[i]+1)
    end

    cond_inputs:zero()
    local outputs = model_D:forward({inputs, cond_inputs})
    -- update confusion_C (add 1 since classes are binary)
    for i = 1,opt.batchSize do
      max, ind = torch.max(outputs[2][i], 1)
      confusion_C:add(ind[1], targets_C[i])
    end

    ----------------------------------------------------------------------
    -- (2) Generated data
    noise_inputs:uniform(-1, 1)
    targets_D:fill(0)
    local sample = dataset[math.random(dataset:size())]
    for i = 1,opt.batchSize do
      cond_inputs[i] = sample[3]:clone()
      targets_C[i] = sample[2]
      if i % 10 == 0 then 
        sample = dataset[math.random(dataset:size())]
      end
    end

    local outputs = model_D:forward({inputs, cond_inputs})
    -- update confusion_D (add 1 since classes are binary)
    for i = 1,opt.batchSize do
      local c
      if outputs[1][i][1] > 0.5 then c = 2 else c = 1 end
      confusion_D:add(c, targets_D[i]+1)
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion_D matrix
  print(confusion_C)
  print(confusion_D)
  testLogger:add{['% mean class accuracy (test set)'] = confusion_D.totalValid * 100}
  confusion_D:zero()
  confusion_C:zero()
end

function adversarial.trainNN(trainData, valData, N, samples)
  local N = N or 25
  if N == 0 then 
    return {}, {}
  end
  -- Get samples from val data
  if samples == nil then
    local noise_inputs 
    if type(opt.noiseDim) == 'number' then
      noise_inputs = torch.Tensor(N, opt.noiseDim)
    else
      noise_inputs = torch.Tensor(N, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
    end
    local cond_inputs 
    if type(opt.condDim) == 'number' then
      cond_inputs = torch.Tensor(N, opt.condDim)
    else
      cond_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
    end
    noise_inputs:uniform(-1, 1)
    local c = 1
    for i = 1,N do
      local sample = valData[i]
      cond_inputs[i] = sample[3]:clone()
    end
  end
  local samples = samples or model_G:forward({noise_inputs, cond_inputs})

  local nearest = {}
  -- Compare samples to train data
  for s = 1,N do
    local min_diff = 1e10
    local sample = samples[s]
    nearest[s] = {}
    nearest[s][1] = sample:clone()
    for d = 1,trainData:size() do
      local data = trainData[d][1]
      local diff = torch.dist(sample, data)
      if diff <= min_diff then
        min_diff = diff
        nearest[s][2] = data:clone()
      end
    end
  end
  -- plot
  local to_plot = {}
  local strs = {}
  for i=1,N do
    to_plot[#to_plot+1] = nearest[i][1]:float()
    to_plot[#to_plot+1] = nearest[i][2]:float()
    strs[#strs+1] = i
    strs[#strs+1] = i
  end
  return to_plot, strs
end
return adversarial
