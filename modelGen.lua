require 'cudnn'
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

   local model = nn.Sequential()
   local factor = getFactor()
   local planes = nPlanes()
   local k = kH()
   model:add(cudnn.SpatialConvolutionUpsample(3+1, planes, k, k, factor, 1))
   if factor > 1 then
      planes = planes / factor
      local t
      if poolType == 'maxout' then
	 t = 1
      elseif poolType == 'poolout' then
	 t = 2
      elseif poolType == 'mixed' then
	 t = torch.random(1,2)
      end
      if t == 1 then
	 model:add(nn.VolumetricMaxPooling(factor, 1, 1))
      elseif t == 2 then
	 model:add(nn.FeatureLPPooling(2,2,2,true))
      end
   end

   local planesIn = planes

   for i=1,nLayers-1 do
      local factor = getFactor()
      local planesOut = nPlanes()
      local k = kH()
      local groups = math.pow(2, torch.random(nGroupsMin, nGroupsMax))
      model:add(cudnn.SpatialConvolutionUpsample(planesIn, planesOut, k, k, factor, groups))
      if factor > 1 then
	 planesOut = planesOut / factor
	 local t
	 if poolType == 'maxout' then
	    t = 1
	 elseif poolType == 'poolout' then
	    t = 2
	 elseif poolType == 'mixed' then
	    t = torch.random(1,2)
	 end
	 if t == 1 then
	    model:add(nn.VolumetricMaxPooling(factor, 1, 1))
	 elseif t == 2 then
	    model:add(nn.FeatureLPPooling(2,2,2,true))
	 end
      end
      planesIn = planesOut
   end
   print(model)
end

generateModelG(2,10,1,64,3,11, 'mixed', 0, 4)
