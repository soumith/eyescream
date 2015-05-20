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
			poolType, nGroupsMin, nGroupsMax, L1Penalty, batchNorm, dropoutType)
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
            print('P ' .. 'MOut ' .. factor)
            model:add(nn.VolumetricMaxPooling(factor, 1, 1))
         elseif t == 2 then
            print('P ' .. 'LPOut ' .. factor)
            model:add(nn.FeatureLPPooling(2,2,2,true))
         end
      end
   end

   local model = nn.Sequential()
   local factor = getFactor()
   local planesOut = torch.random(1,5) * 16
   local k = kH()
   print('C ' .. 4 .. '->' .. planesOut .. '/' .. 1 .. ' ' .. k .. 'x' .. k)
   model:add(cudnn.SpatialConvolutionUpsample(3+1, planesOut, k, k, 1, 1))
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
         -- print('here: ', pow, groups, planesIn, planesOut)
      end
      print('C ' .. planesIn .. '->' .. planesOut .. '/' .. groups .. ' ' .. k .. 'x' .. k)
      model:add(cudnn.SpatialConvolutionUpsample(planesIn, planesOut, k, k, 1, groups))
      pool(model, factor)
      planesOut = planesOut / factor
      planesIn = planesOut
   end
   print('C ' .. planesIn .. '->' .. 3 .. '/' .. 1 .. ' ' .. k .. 'x' .. k)
   model:add(cudnn.SpatialConvolutionUpsample(planesIn, 3, k, k, 1, 1))

   -- print(model)
   model=model:cuda()
   local input = torch.zeros(10,4,32,32):cuda()
   model:forward(input)
   return model
end

modelG = generateModelG(2,5,1,64,3,11, 'mixed', 0, 4)
