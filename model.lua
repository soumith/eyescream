paths.dofile('modelGen.lua')
----------------------------------------------------------------------
if opt.network ~= '' then
  print('<trainer> reloading previously trained network: ' .. opt.network)
  local tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
  print('Discriminator network:')
  print(model_D)
  print('Generator network:')
  print(model_G)
else
   -- define D network to train
   model_D = generateModelD(2,6,1,64,3,11, 'mixed', 0, 4, 2)
   -- define G network to train
   print('Generator network:')
   model_G = generateModelG(2,5,1,64,3,11, 'mixed', 0, 4, 2)
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   print(model_G)
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

model_D:cuda()
model_G:cuda()
criterion:cuda()


-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

local freeParams = function(m)
   local list = m:listModules()
   local p = 0
   for k,v in pairs(list) do
      p = p + (v.weight and v.weight:nElement() or 0)
      p = p + (v.bias and v.bias:nElement() or 0)
   end
   return p
end
print('\nNumber of free parameters in D: ' .. freeParams(model_D))
print('Number of free parameters in G: ' .. freeParams(model_G) .. '\n')
