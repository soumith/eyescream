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
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')
  for t = 1,dataset:size(),opt.batchSize do

    img_sz = opt.geometry[1]*opt.geometry[2]*opt.geometry[3]
    local inputs = torch.Tensor(opt.batchSize, img_sz)
    local targets = torch.Tensor(opt.batchSize)
    local noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim) 
    local cond_inputs = torch.Tensor(opt.batchSize, opt.condDim)

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      --debugger.enter()

      --  forward pass
      local outputs = model_D:forward({inputs, cond_inputs})
      local f = criterion:forward(outputs, targets)

      -- backward pass 
      local df_do = criterion:backward(outputs, targets)
      model_D:backward({inputs, cond_inputs}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end
      -- update confusion (add 1 since classes are binary)
      for i = 1,opt.batchSize do
        local c
        if outputs[i][1] > 0.5 then c = 2 else c = 1 end
        confusion:add(c, targets[i]+1)
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
--      debugger.enter()

      -- forward pass
      local samples = model_G:forward({noise_inputs, cond_inputs})
      local outputs = model_D:forward({samples, cond_inputs})
      local f = criterion:forward(outputs, targets)

      --  backward pass
      local df_samples = criterion:backward(outputs, targets)
      model_D:backward({samples, cond_inputs}, df_samples)
      local df_do = model_D.modules[1].gradInput[1]
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
      targets:ones(targets:size())
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
        -- load new sample
        local sample = dataset[i]
        inputs[k] = sample[1]:reshape(img_sz)
        cond_inputs[k] = sample[3]:reshape(opt.condDim)
        k = k + 1
      end

      optim.sgd(fevalD, parameters_D, sgdState_D)

      -- (1.2) Sampled data
      noise_inputs:uniform(-1, 1)
      for i = 1,opt.batchSize do
        local idx = math.random(dataset:size())
        local sample = dataset[idx]
        cond_inputs[i] = sample[3]:reshape(opt.condDim)
      end
      local samples = model_G:forward({noise_inputs, cond_inputs})
      inputs:copy(samples)
      targets:zeros(targets:size())

      optim.sgd(fevalD, parameters_D, sgdState_D)
    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:uniform(-1, 1)
    for i = 1,opt.batchSize do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      cond_inputs[i] = sample[3]:reshape(opt.condDim)
    end
    targets:ones(targets:size())
    optim.sgd(fevalG, parameters_G, sgdState_G)

    -- disp progress
    xlua.progress(t, dataset:size())
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()

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
  local time = sys.clock()
  local img_sz = opt.geometry[1]*opt.geometry[2]*opt.geometry[3]

  local noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)
  local cond_inputs = torch.Tensor(opt.batchSize, opt.condDim)
  local inputs = torch.Tensor(opt.batchSize, img_sz)
  print('\n<trainer> on testing Set:')
  for t = 1,dataset:size(),opt.batchSize do
    -- display progress
    xlua.progress(t, dataset:size())

    ----------------------------------------------------------------------
    -- (1) Real data
    local targets = torch.ones(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      local sample = dataset[i]
      inputs[k] = sample[1]:reshape(img_sz)
      cond_inputs[k] = sample[3]:reshape(opt.condDim)
      k = k + 1
    end
    local preds = model_D:forward({inputs, cond_inputs}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end

    ----------------------------------------------------------------------
    -- (2) Generated data
    noise_inputs:uniform(-1, 1)
    local c = 1
    local sample = dataset[math.random(dataset:size())]
    for i = 1,opt.batchSize do
      cond_inputs[i] = sample[3]:reshape(opt.condDim)
      if i % 10 == 0 then 
        sample = dataset[math.random(dataset:size())]
      end
    end
    local samples = model_G:forward({noise_inputs, cond_inputs})
    local targets = torch.zeros(opt.batchSize)
    local preds = model_D:forward({samples, cond_inputs}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()

  return cond_inputs
end

return adversarial
