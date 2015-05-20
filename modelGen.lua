require 'cudnn'
paths.dofile('layers/cudnnSpatialConvolutionUpsample.lua')
require 'fbcunn'

-- square kernels.
-- pool size == stride
-- poolType = max,l2,avg,maxout,poolout,mixed
-- poolOrder = 0/1, 0 = pool + upsampling. 1 = upsampling + pooling
-- minHidden/maxHidden - linear layer hidden units
-- dropoutType - 0,1,2. 0 = no dropout, 1 = Dropout, 2 = SpatialDropout + Dropout
-- L1Penalty = 0,1 (enabled or disabled)
-- batchNorm = 0,1,2 (0 = disabled, 1 = before relu, 2 = after relu)

-- ngroups, 0,4
function generateModelG(minLayers, maxLayers, minPlanes, maxPlanes, minKH, maxKH,
			poolType, nGroupsMin, nGroupsMax, dropoutType, L1Penalty, batchNorm)
   local desc = ''
   if dropoutType > 0 then local dd = torch.random(1,2); if dd == 1 then dropoutType = 0 end end
   local nLayers = torch.random(minLayers, maxLayers)
   local function nPlanes()
      return torch.random(minPlanes, maxPlanes) * 16 -- planes are always a multiple of 16
   end
   local function kH()
      return math.floor((torch.random(minKH, maxKH)/2)) * 2 + 1 -- odd kernel size
   end
   local function getFactor()
      if poolType == 'mixed' or poolType == 'maxout' or poolType == 'poolout' then
	 return torch.random(1,2)
      else
	 return 1
      end
   end

   local function pool(model, factor)
      if factor > 1 then
         local t
         if poolType == 'maxout' then t = 1
         elseif poolType == 'poolout' then t = 2
         elseif poolType == 'mixed' then t = torch.random(1,2) end
         if t == 1 then
            desc = desc .. '___P_' .. 'MOut_' .. factor
            model:add(nn.VolumetricMaxPooling(factor, 1, 1))
         elseif t == 2 then
            desc = desc .. '___P_' .. 'LPOut_' .. factor
            model:add(nn.FeatureLPPooling(2,2,2,true))
         end
      end
   end

   local model = nn.Sequential()
   model:add(nn.JoinTable(2,2))
   local factor = getFactor()
   local planesOut = torch.random(1,5) * 16
   local k = kH()
   desc = desc .. '___C_' .. 4 .. '_' .. planesOut .. '_g' .. 1 .. '_' .. k .. 'x' .. k
   model:add(cudnn.SpatialConvolutionUpsample(3+1, planesOut, k, k, 1, 1))
   model:add(cudnn.ReLU(true))
   pool(model, factor)
   planesOut = planesOut / factor

   local planesIn = planesOut

   for i=1,nLayers-2 do
      local factor = getFactor()
      local planesOut = nPlanes()
      local k = kH()
      local groups = 13
      while planesIn % groups ~= 0 or planesOut % groups ~= 0 do
         local pow
         if planesOut > 256 or planesIn > 256 then
            pow = torch.random(2, nGroupsMax)
         else
            pow = torch.random(nGroupsMin, nGroupsMax)
         end
         groups = math.pow(2, pow)
      end
      desc = desc .. '___C_' .. planesIn .. '_' .. planesOut .. '_g' .. groups .. '_' .. k .. 'x' .. k
      model:add(cudnn.SpatialConvolutionUpsample(planesIn, planesOut, k, k, 1, groups))
      model:add(cudnn.ReLU(true))
      if dropoutType == 2 then
         local dp = torch.random(1,2)
         if dp == 2 then
            model:add(nn.SpatialDropout(0.5))
            desc = desc .. '___SDrop ' .. 0.5
         end
      end
      pool(model, factor)
      planesOut = planesOut / factor
      planesIn = planesOut
   end
   desc = desc .. '___C_' .. planesIn .. '_' .. 3 .. '_g' .. 1 .. '_' .. k .. 'x' .. k
   model:add(cudnn.SpatialConvolutionUpsample(planesIn, 3, k, k, 1, 1))
   model:add(cudnn.ReLU(true))

   model=model:cuda()
   local input1 = torch.zeros(10,3,32,32):cuda()
   local input2 = torch.zeros(10,1,32,32):cuda()
   model:forward({input1, input2})
   return model, desc
end

function generateModelD(minLayers, maxLayers, minPlanes, maxPlanes, minKH, maxKH,
                 poolType, nGroupsMin, nGroupsMax, dropoutType, batchNorm)
   local iH = opt.geometry[2]
   local desc = ''
   if dropoutType > 0 then local dd = torch.random(1,2); if dd == 1 then dropoutType = 0 end end
   local nLayers = torch.random(minLayers, maxLayers)
   local function nPlanes()
      return torch.random(minPlanes, maxPlanes) * 16 -- planes are always a multiple of 16
   end
   local function kH()
      return math.floor((torch.random(minKH, maxKH)/2)) * 2 + 1 -- odd kernel size
   end
   local function pool(model,planes)
      if torch.random(1,2) == 1 then return end
      if math.floor(iH / 2) < 1 then return true end
      local t
      if poolType == 'max' then t = 3
      elseif poolType == 'lp' then t = 4
      elseif poolType == 'avg' then t = 5
      elseif poolType == 'mixed' then t = torch.random(3,5) end
      if t == 3 then
         desc = desc .. '___P_' .. 'Max_' .. 2
         model:add(cudnn.SpatialMaxPooling(2,2,2,2))
      elseif t == 4 then
         desc = desc .. '___P_' .. 'LP_' .. 2
         model:add(nn.SpatialLPPooling(planes,2,2,2,2,2))
      elseif t == 5 then
         desc = desc .. '___P_' .. 'Avg_' .. 2
         model:add(cudnn.SpatialAveragePooling(2,2,2,2))
      end
      iH = math.floor(iH / 2)
   end

   local model = nn.Sequential()
   model:add(nn.CAddTable())

   local planesOut = torch.random(1,5) * 16
   local k = kH()
   desc = desc .. '___C_' .. 3 .. '_' .. planesOut .. '_g' .. 1 .. '_' .. k .. 'x' .. k
   model:add(cudnn.SpatialConvolution(3, planesOut, k, k, 1, 1))
   model:add(cudnn.ReLU(true))
   iH = iH - k + 1
   pool(model, planesOut)
   assert(iH >= 1)

   local planesIn = planesOut

   for i=1,nLayers-2 do
      local planesOut = nPlanes()
      local k = kH()
      local groups = 13
      while planesIn % groups ~= 0 or planesOut % groups ~= 0 do
         local pow
         if planesOut > 256 or planesIn > 256 then
            pow = torch.random(2, nGroupsMax)
         else
            pow = torch.random(nGroupsMin, nGroupsMax)
         end
         groups = math.pow(2, pow)
      end
      if (iH - k + 1) < 1 then break end
      iH = iH - k + 1
      assert(iH >= 1)
      desc = desc .. '___C_' .. planesIn .. '_' .. planesOut .. '_g' .. groups .. '_' .. k .. 'x' .. k
      model:add(cudnn.SpatialConvolution(planesIn, planesOut, k, k, 1, 1, 0, 0, groups))
      model:add(cudnn.ReLU(true))
      planesIn = planesOut
      if pool(model, planesOut) then break end
      assert(iH >= 1)
   end
   model:add(nn.View(planesIn * iH * iH):setNumInputDims(3))
   desc = desc .. '___L ' .. planesIn * iH * iH .. '_' .. 1
   model:add(nn.Linear(planesIn * iH * iH, 1))
   model:add(nn.ReLU())
   model:add(nn.Sigmoid())

   -- print(model)
   model=model:cuda()
   local input1 = torch.zeros(10,3,32,32):cuda()
   local input2 = torch.zeros(10,3,32,32):cuda()
   model:forward({input1, input2})
   return model, desc
end
