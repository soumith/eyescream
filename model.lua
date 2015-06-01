paths.dofile('modelGen.lua')
----------------------------------------------------------------------
local freeParams = function(m)
   local list = m:listModules()
   local p = 0
   for k,v in pairs(list) do
      p = p + (v.weight and v.weight:nElement() or 0)
      p = p + (v.bias and v.bias:nElement() or 0)
   end
   return p
end
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
elseif opt.model == 'emily' then -- TODO: add as option
   local nplanes = 64
   model_D = nn.Sequential()
   model_D:add(nn.CAddTable())
   model_D:add(nn.SpatialConvolution(3, nplanes, 5, 5)) --28 x 28
   model_D:add(nn.ReLU())
   model_D:add(nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2))
   local sz = math.floor( ( (opt.fineSize - 5 + 1) - 5) / 2 + 1)
   model_D:add(nn.View(nplanes*sz*sz))
   model_D:add(nn.ReLU())
   model_D:add(nn.Linear(nplanes*sz*sz, 1))
   model_D:add(nn.Sigmoid())
   local nplanes = 128
   model_G = nn.Sequential()
   model_G:add(nn.JoinTable(2, 2))
   model_G:add(cudnn.SpatialConvolutionUpsample(3+1, nplanes, 7, 7, 1))
   model_G:add(nn.ReLU())
   model_G:add(cudnn.SpatialConvolutionUpsample(nplanes, nplanes, 7, 7, 1))
   model_G:add(nn.ReLU())
   model_G:add(cudnn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 1))
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
elseif opt.model == 'large' then
   print('Generator network (good):')
   desc_G = '___JT22___C_4_64_g1_7x7___R__BN___C_64_368_g4_7x7___R__BN___SDrop 0.5___C_368_128_g4_7x7___R__BN___P_LPOut_2___C_64_224_g2_5x5___R__BN___SDrop 0.5___C_224_3_g1_7x7__BNA'
   model_G = nn.Sequential()
   model_G:add(nn.JoinTable(2, 2))
   model_G:add(cudnn.SpatialConvolutionUpsample(3+1, 64, 7, 7, 1, 1)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(64, nil, nil, false))
   model_G:add(cudnn.SpatialConvolutionUpsample(64, 368, 7, 7, 1, 4)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(368, nil, nil, false))
   model_G:add(nn.SpatialDropout(0.5))
   model_G:add(cudnn.SpatialConvolutionUpsample(368, 128, 7, 7, 1, 4)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(128, nil, nil, false))
   model_G:add(nn.FeatureLPPooling(2,2,2,true))
   model_G:add(cudnn.SpatialConvolutionUpsample(64, 224, 5, 5, 1, 2)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(224, nil, nil, false))
   model_G:add(nn.SpatialDropout(0.5))
   model_G:add(cudnn.SpatialConvolutionUpsample(224, 3, 7, 7, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(3, nil, nil, false))
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   print(desc_G)

   desc_D = '___CAdd___C_3_48_g1_3x3___R___C_48_448_g4_5x5___R___C_448_416_g16_7x7___R___V_166400___L 166400_1___Sig'
   model_D = nn.Sequential()
   model_D:add(nn.CAddTable())
   model_D:add(cudnn.SpatialConvolution(3, 48, 3, 3))
   model_D:add(cudnn.ReLU(true))
   model_D:add(cudnn.SpatialConvolution(48, 448, 5, 5, 1, 1, 0, 0, 4))
   model_D:add(cudnn.ReLU(true))
   model_D:add(cudnn.SpatialConvolution(448, 416, 7, 7, 1, 1, 0, 0, 16))
   model_D:add(cudnn.ReLU())
   model_D:cuda()
   local dummy_input = torch.zeros(opt.batchSize, 3, opt.fineSize, opt.fineSize):cuda()
   local out = model_D:forward({dummy_input, dummy_input})
   local nElem = out:nElement() / opt.batchSize
   model_D:add(nn.View(nElem):setNumInputDims(3))
   model_D:add(nn.Linear(nElem, 1))
   model_D:add(nn.Sigmoid())
   model_D:cuda()
   print(desc_D)
elseif opt.model == 'autogen' then
   -- define G network to train
   print('Generator network:')
   model_G,desc_G = generateModelG(3,5,128,512,3,7, 'mixed', 0, 4, 2, true)
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   print(desc_G)
   print(model_G)
   local trygen = 1
   local poolType = 'none'
   -- if torch.random(1,2) == 1 then poolType = 'none' end
   repeat
      trygen = trygen + 1
      if trygen == 500 then error('Could not find a good D model') end
      -- define D network to train
      print('Discriminator network:')
      model_D,desc_D = generateModelD(2,6,64,512,3,7, poolType, 0, 4, 2)
      print(desc_D)
      print(model_D)
   until (freeParams(model_D) < freeParams(model_G))
      and (freeParams(model_D) > freeParams(model_G) / 10)
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

model_D:cuda()
model_G:cuda()
criterion:cuda()


-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

print('\nNumber of free parameters in D: ' .. freeParams(model_D))
print('Number of free parameters in G: ' .. freeParams(model_G) .. '\n')
