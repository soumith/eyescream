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
    local noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      --  forward pass
      local outputs = model_D:forward(inputs)
      local f = criterion_D:forward(outputs[1], targets_D)

      -- backward pass 
      local dD_do = criterion_D:backward(outputs[1], targets_D)
      local dC_do 
      if targets_C[1] == -1 then -- dont back prop wrt these targets
        dC_do = torch.zeros(outputs[2]:size())
      else
        dC_do = criterion_C:backward(outputs[2], targets_C) 
      end
      model_D:backward(inputs, {dD_do, dC_do})

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

      -- update confusion_C 
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
      local samples = model_G:forward(noise_inputs)
      local outputs = model_D:forward(samples)
      local f = criterion_D:forward(outputs[1], targets_D)

      --  backward pass
      local dD_samples = criterion_D:backward(outputs[1], targets_D)
      local dC_samples = torch.zeros(outputs[2]:size())
      model_D:backward({samples, nil}, {dD_samples, dC_samples})
      local df_do = model_D.gradInput
      model_G:backward(noise_inputs, df_do)

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
        local input = sample[1]:clone()
        inputs[k] = input
        targets_C[k] = sample[2]
        k = k + 1
      end

      optim.sgd(fevalD, parameters_D, sgdState_D)

      -- (1.2) Sampled data
      noise_inputs:uniform(-1, 1)
      local samples = model_G:forward(noise_inputs)
      inputs:copy(samples)
      targets_D:fill(0)
      targets_C:fill(-1)

      optim.sgd(fevalD, parameters_D, sgdState_D)
    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:uniform(-1, 1)
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
    local filename = paths.concat(opt.save, 'adversarial.net')
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

  print('\n<trainer> on testing Set:')
  for t = 1,dataset:size(),opt.batchSize do
    -- disp progress
    xlua.progress(t, dataset:size())

    -- (1) Real data
    local inputs = torch.Tensor(opt.batchSize,opt.geometry[1],opt.geometry[2], opt.geometry[3])
    local targets_D = torch.ones(opt.batchSize)
    local targets_C = torch.ones(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      inputs[k] = sample[1]:clone()
      targets_C[k] = sample[2]
      k = k + 1
    end
    local outputs = model_D:forward(inputs)
    -- update confusion_D 
    for i = 1,opt.batchSize do
      local c
      if outputs[1][i][1] > 0.5 then c = 2 else c = 1 end
      confusion_D:add(c, targets_D[i] + 1)
    end
    -- update confusion_C 
    for i = 1,opt.batchSize do
      max, ind = torch.max(outputs[2][i], 1)
      confusion_C:add(ind[1], targets_C[i])
    end

    -- (2) Generated data
    local noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim):uniform(-1, 1)
    local inputs = model_G:forward(noise_inputs)
    local targets = torch.zeros(opt.batchSize)
    local outputs = model_D:forward(inputs)
    for i = 1,opt.batchSize do
      local c
      if outputs[1][i][1] > 0.5 then c = 2 else c = 1 end
      confusion_D:add(c, targets[i] + 1)
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion_D matrix
  print(confusion_D)
  print(confusion_C)
  testLogger:add{['% mean class accuracy (test set)'] = confusion_D.totalValid * 100}
  confusion_D:zero()
  confusion_C:zero()
end

function adversarial.trainNN(trainData, valData, N, samples)
  local N = N or 25
  -- Get samples from val data
  local noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim):uniform(-1, 1)
  local samples = samples or model_G:forward(noise_inputs)

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
