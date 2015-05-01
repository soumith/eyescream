require 'torch'
require 'nn'
require 'cunn'
require 'nnx'
require 'optim'
require 'pl'
require 'paths'

local adversarial = {}

-- training function
function adversarial.train(dataset)
  -- epoch tracker
  epoch = epoch or 1

  -- local vars
  local time = sys.clock()

  -- do one epoch
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D[1].learningRate .. ', momentum = ' .. sgdState_D[1].momentum .. ']')
  for t = 1,dataset:size(),opt.batchSize do

    local inputs = {}
    for i = 1,opt.numres do
      inputs[i] = torch.Tensor(opt.batchSize, opt.geometry[i][1], opt.geometry[i][2], opt.geometry[i][3])
    end
    local targets = torch.Tensor(opt.batchSize)
    local noise_inputs = {}
    for i = 1,opt.numres do
      noise_inputs[i] = torch.Tensor(opt.batchSize, opt.noiseDim[i][1], opt.noiseDim[i][2], opt.noiseDim[i][3])
    end
    local cond_inputs = {}
    for i = 1,opt.numres do
      cond_inputs[i] = torch.Tensor(opt.batchSize, opt.condDim[i][1], opt.condDim[i][2], opt.condDim[i][3])
    end
    local generated_samples = {}
    for i = 1,opt.numres-1 do
      generated_samples[i] = torch.Tensor(opt.batchSize, opt.geometry[i+1][1], opt.geometry[i+1][2], opt.geometry[i+1][3])
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = {}
    for i = 1,opt.numres do
      fevalD[i] = function(x)
        collectgarbage()
        if x ~= parameters_D[i] then -- get new parameters
          parameters_D[i]:copy(x)
        end

        gradParameters_D[i]:zero() -- reset gradients

        --  forward pass
        local outputs = model_D[i]:forward({inputs[i], cond_inputs[i]})
        local f = criterion:forward(outputs, targets)

        -- backward pass 
        local df_do = criterion:backward(outputs, targets)
        model_D[i]:backward({inputs[i], cond_inputs[i]}, df_do)

        -- penalties (L1 and L2):
        if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
          local norm,sign= torch.norm,torch.sign
          -- Loss:
          f = f + opt.coefL1 * norm(parameters_D[i],1)
          f = f + opt.coefL2 * norm(parameters_D[i],2)^2/2
          -- Gradients:
          gradParameters_D[i]:add( sign(parameters_D[i]):mul(opt.coefL1) + parameters_D[i]:clone():mul(opt.coefL2) )
        end
        -- update confusion (add 1 since classes are binary)
        for b = 1,opt.batchSize do
          local c
          if outputs[b][1] > 0.5 then c = 2 else c = 1 end
          confusion[i]:add(c, targets[b]+1)
        end

        return f,gradParameters_D[i]
      end
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = {}
    for i=1,opt.numres do
      fevalG[i] = function(x)
        collectgarbage()
        if x ~= parameters_G[i] then -- get new parameters
          parameters_G[i]:copy(x)
        end
        
        gradParameters_G[i]:zero() -- reset gradients
  --      debugger.enter()

        -- forward pass
        local samples = model_G[i]:forward({noise_inputs[i], cond_inputs[i]})
        local outputs = model_D[i]:forward({samples, cond_inputs[i]})
        local f = criterion:forward(outputs, targets)

        --  backward pass
        local df_samples = criterion:backward(outputs, targets)
        model_D[i]:backward({samples, cond_inputs[i]}, df_samples)
        local df_do = model_D[i].gradInput[1]
        model_G[i]:backward({noise_inputs[i], cond_inputs[i]}, df_do)

        -- penalties (L1 and L2):
        if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
          local norm,sign= torch.norm,torch.sign
          -- Loss:
          f = f + opt.coefL1 * norm(parameters_D,1)
          f = f + opt.coefL2 * norm(parameters_D,2)^2/2
          -- Gradients:
          gradParameters_G[i]:add( sign(parameters_G[i]):mul(opt.coefL1) + parameters_G[i]:clone():mul(opt.coefL2) )
        end

        return f,gradParameters_G[i]
      end
    end

    for r = 1,opt.numres do
      ----------------------------------------------------------------------
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      for k=1,opt.K do
        -- (1.1) Real data 
        targets:ones(targets:size())
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
          -- load new sample
          local sample = dataset[r][i]
          inputs[r][k] = sample[1]:clone()
          if true then --r == 1 then
            cond_inputs[r][k] = sample[3]:clone()
          else
            cond_inputs[r][k] = generated_samples[r-1][k]:clone()
          end
          k = k + 1
        end

        optim.sgd(fevalD[r], parameters_D[r], sgdState_D[r])

        -- (1.2) Sampled data
        noise_inputs[r]:uniform(-1, 1)
        for i = 1,opt.batchSize do
          local idx = math.random(dataset:size())
          local sample = dataset[r][idx]
          if true then --r == 1 then
            cond_inputs[r][i] = sample[3]:clone()
          else
            cond_inputs[r][i] = generated_samples[r-1][i]:clone()
          end
        end
        local samples = model_G[r]:forward({noise_inputs[r], cond_inputs[r]})
        inputs[r]:copy(samples)
        if r < opt.numres then
          samples:reshape(samples, opt.batchSize, opt.geometry[r][1], opt.geometry[r][2], opt.geometry[r][3])
          local tmp = torch.add(samples, cond_inputs[r])
          for i = 1,opt.batchSize do
            for c = 1,opt.geometry[r][1] do
              generated_samples[r][i][c] = image.scale(tmp[i][c]:float(), opt.fineSize[r+1], opt.fineSize[r+1])
            end
          end
        end
        targets:zeros(targets:size())

        optim.sgd(fevalD[r], parameters_D[r], sgdState_D[r])
      end -- end for K

      ----------------------------------------------------------------------
      -- (2) Update G network: maximize log(D(G(z)))
      noise_inputs[r]:uniform(-1, 1)
      for i = 1,opt.batchSize do
        local idx = math.random(dataset:size())
        local sample = dataset[r][idx]
        if true then --r == 1 then
          cond_inputs[r][i] = sample[3]:clone()
        else
          cond_inputs[r][i] = generated_samples[r-1][i]:clone()
        end
      end
      targets:ones(targets:size())
      optim.sgd(fevalG[r], parameters_G[r], sgdState_G[r])

      -- disp progress
      xlua.progress(t, dataset:size())
    end -- end for loop over dataset

  end
  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  for i = 1,opt.numres do
    print(confusion[i])
    trainLogger:add{['% mean class accuracy (train set)'] = confusion[i].totalValid * 100}
    confusion[i]:zero()
  end

  -- save/log current net
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, 'multires_conditional_adversarial.net')
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
  local time = sys.clock()

  local inputs = {}
  for i = 1,opt.numres do
    inputs[i] = torch.Tensor(opt.batchSize, opt.geometry[i][1], opt.geometry[i][2], opt.geometry[i][3])
  end
  local targets = torch.Tensor(opt.batchSize)
  local noise_inputs = {}
  for i = 1,opt.numres do
    noise_inputs[i] = torch.Tensor(opt.batchSize, opt.noiseDim[i][1], opt.noiseDim[i][2], opt.noiseDim[i][3])
  end
  local cond_inputs = {}
  for i = 1,opt.numres do
    cond_inputs[i] = torch.Tensor(opt.batchSize, opt.condDim[i][1], opt.condDim[i][2], opt.condDim[i][3])
  end
  local generated_samples = {}
  for i = 1,opt.numres-1 do
    generated_samples[i] = torch.Tensor(opt.batchSize, opt.geometry[i+1][1], opt.geometry[i+1][2], opt.geometry[i+1][3])
  end

  print('\n<trainer> on testing Set:')
  for t = 1,dataset:size(),opt.batchSize do
    -- display progress
    xlua.progress(t, dataset:size())

    for r = 1,opt.numres do
      ----------------------------------------------------------------------
      -- (1) Real data
      local targets = torch.ones(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
        local idx = math.random(dataset:size())
        local sample = dataset[r][idx]
        inputs[r][k] = sample[1]:clone()
        if true then --r == 1 then
          cond_inputs[r][k] = sample[3]:clone()
        else
          cond_inputs[r][k] = generated_samples[r-1][k]:clone()
        end
        k = k + 1
      end
      local preds = model_D[r]:forward({inputs[r], cond_inputs[r]}) -- get predictions from D
      -- add to confusion matrix
      for i = 1,opt.batchSize do
        local c
        if preds[i][1] > 0.5 then c = 2 else c = 1 end
        confusion[r]:add(c, targets[i] + 1)
      end

      ----------------------------------------------------------------------
      -- (2) Generated data
      noise_inputs[r]:uniform(-1, 1)
      for i = 1,opt.batchSize do
        local sample = dataset[r][math.random(dataset:size())]
        if r == 1 then
          cond_inputs[r][i] = sample[3]:clone()
        else
          cond_inputs[r][i] = generated_samples[r-1][i]:clone()
        end
      end
      local samples = model_G[r]:forward({noise_inputs[r], cond_inputs[r]})
      local targets = torch.zeros(opt.batchSize)
      local preds = model_D[r]:forward({samples, cond_inputs[r]}) -- get predictions from D
      if r < opt.numres then
        samples:reshape(samples, opt.batchSize, opt.geometry[r][1], opt.geometry[r][2], opt.geometry[r][3])
        local tmp = torch.add(samples, cond_inputs[r])
        for i = 1,opt.batchSize do
          for c = 1,opt.geometry[r][1] do
            generated_samples[r][i][c] = image.scale(tmp[i][c]:float(), opt.fineSize[r+1], opt.fineSize[r+1])
          end
        end
      end
      
      -- add to confusion matrix
      for i = 1,opt.batchSize do
        local c
        if preds[i][1] > 0.5 then c = 2 else c = 1 end
        confusion[r]:add(c, targets[i] + 1)
      end
    end -- end loop over dataset
  end

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  for i = 1,opt.numres do
    print(confusion[i])
    testLogger:add{['% mean class accuracy (test set)'] = confusion[i].totalValid * 100}
    confusion[i]:zero()
  end

  return cond_inputs
end

function adversarial.trainNN(trainData, valData, N)
  local N = N or 25
  -- Get samples from val data
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
  local sample = valData[math.random(valData:size())]
  for i = 1,N do
    cond_inputs[i] = sample[3]:clone()
    if i % 10 == 0 then 
      sample = valData[math.random(valData:size())]
    end
  end
  local samples = model_G[r]:forward({noise_inputs, cond_inputs})

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
