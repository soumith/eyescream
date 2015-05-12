----------------------------------------------------------------------
-- define D network to train
local nplanes = 128
model_D = nn.Sequential()
model_D:add(nn.CAddTable())
model_D:add(nn.SpatialConvolution(3, nplanes, 5, 5)) --28 x 28
model_D:add(nn.ReLU())
model_D:add(nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2))
model_D:add(nn.View(nplanes*12*12):setNumInputDims(3))
model_D:add(nn.ReLU())
model_D:add(nn.Linear(nplanes*12*12, 1))
model_D:add(nn.Sigmoid())
----------------------------------------------------------------------
-- define G network to train
local nplanes = 128
model_G = nn.Sequential()
model_G:add(nn.JoinTable(2, 2))
model_G:add(nn.SpatialConvolution(3+1, nplanes, 7, 7, 2,2,3)) -- gives 16x16
model_G:add(nn.ReLU())
-- model_G:add(nn.SpatialConvolutionUpsample(nplanes, nplanes, 5, 5, 2)) -- 3 color channels + conditional
-- model_G:add(nn.ReLU())
model_G:add(nn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 2)) -- 3 color channels + conditional
model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]):setNumInputDims(3))

if opt.network ~= '' then
  print('<trainer> reloading previously trained network: ' .. opt.network)
  local tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

model_D:cuda()
model_G:cuda()
criterion:cuda()


-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

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
