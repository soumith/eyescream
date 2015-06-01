require 'torch'
require 'optim'
require 'paths'

local adversarial = {}

-- reusable buffers
local targets      = torch.CudaTensor(opt.batchSize)
local inputs       = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))               -- original full-res image - low res image
local cond_inputs  = torch.CudaTensor(opt.batchSize, unpack(opt.condDim))  -- low res image blown up and differenced from original
local noise_inputs = torch.CudaTensor(opt.batchSize, unpack(opt.noiseDim)) -- pure noise
local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()

-- training function
function adversarial.train(inputs_all, inputs_all2)
   local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
   local err_G, err_D

   -- inputs_all = {diff, label, coarse, fine}
   inputs:copy(inputs_all[1])
   cond_inputs:copy(inputs_all[3])

   -- create closure to evaluate f(X) and df/dX of discriminator
   local fevalD = function(x)
      collectgarbage()
      gradParameters_D:zero() -- reset gradients

      --  forward pass
      local outputs = model_D:forward({inputs, cond_inputs})
      err_D = criterion:forward(outputs, targets)

      -- backward pass
      local df_do = criterion:backward(outputs, targets)
      model_D:backward({inputs, cond_inputs}, df_do)

      -- update confusion (add 1 since classes are binary)
      outputs[outputs:gt(0.5)] = 2
      outputs[outputs:le(0.5)] = 1
      confusion:batchAdd(outputs, targets:clone():add(1))

      return err_D,gradParameters_D
   end
   ----------------------------------------------------------------------
   -- create closure to evaluate f(X) and df/dX of generator
   local fevalG = function(x)
      collectgarbage()
      gradParameters_G:zero() -- reset gradients

      -- forward pass
      local hallucinations = model_G:forward({noise_inputs, cond_inputs})
      local outputs = model_D:forward({hallucinations, cond_inputs})
      err_G = criterion:forward(outputs, targets)

      --  backward pass
      local df_hallucinations = criterion:backward(outputs, targets)
      model_D:backward({hallucinations, cond_inputs}, df_hallucinations)
      local df_do = model_D.modules[1].gradInput[1]
      model_G:backward({noise_inputs, cond_inputs}, df_do)

      return err_G,gradParameters_G
   end
   ----------------------------------------------------------------------
   -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
   for k = 1, opt.K do
      assert (opt.batchSize % 2 == 0)
      -- (1.1) Real data is in {inputs, cond_inputs}
      targets:fill(1)
      -- (1.2) Sampled data
      noise_inputs:uniform(-1, 1)
      local inps = {noise_inputs, cond_inputs}
      local hallucinations = model_G:forward(inps)
      assert(hallucinations:size(1) == opt.batchSize)
      assert(hallucinations:size(2) == 3)
      assert(hallucinations:nElement() == inputs:nElement())
      -- print(#hallucinations)
      -- print(#inputs)
      inputs:narrow(1, 1, opt.batchSize / 2):copy(hallucinations:narrow(1, 1, opt.batchSize / 2))
      targets:narrow(1, 1, opt.batchSize / 2):fill(0)
      optim.sgd(fevalD, parameters_D, sgdState_D)
   end -- end for K
   ----------------------------------------------------------------------
   -- (2) Update G network: maximize log(D(G(z)))
   noise_inputs:uniform(-1, 1)
   targets:fill(1)
   cond_inputs:copy(inputs_all2[3])
   optim.sgd(fevalG, parameters_G, sgdState_G)
   batchNumber = batchNumber + 1
   cutorch.synchronize(); collectgarbage();
   -- xlua.progress(batchNumber, opt.epochSize)
   print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f Err_G %.4f Err_D %.4f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime, err_G, err_D))
   dataTimer:reset()
end

-- test function
function adversarial.test(inputs_all)
   -- (1) Real data
   targets:fill(1)
   inputs:copy(inputs_all[1])
   cond_inputs:copy(inputs_all[3])

   local outputs = model_D:forward({inputs, cond_inputs}) -- get predictions from D

   -- add to confusion matrix
   outputs[outputs:gt(0.5)] = 2
   outputs[outputs:le(0.5)] = 1
   confusion:batchAdd(outputs, targets:clone():add(1))
   ----------------------------------------------------------------------
   -- (2) Generated data
   noise_inputs:uniform(-1, 1)
   local samples = model_G:forward({noise_inputs, cond_inputs})
   targets:fill(0)
   local outputs = model_D:forward({samples, cond_inputs})
end

return adversarial
