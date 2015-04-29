require 'torch'

local image = {}

function image.normalize(data, mean_, std_)
  local mean = mean or data:mean(1)
  local std = std_ or data:std(1, true)
  local eps = 1e-7
  for i=1,data:size(1) do
    data[i]:add(-1, mean)
    data[i]:cdiv(std + eps)
  end
  return mean, std
end

function image.normalizeGlobal(data, mean_, std_)
  local std = std_ or data:std()
  local mean = mean_ or data:mean()
  data:add(-mean)
  data:mul(1/std)
  return mean, std
end

function image.contrastNormalize(data, new_min, new_max)
  local old_max = data:max(1)
  local old_min = data:min(1)
  local eps = 1e-7
  for i=1,data:size(1) do
    data[i]:add(-1, old_min)
    data[i]:mul(new_max - new_min)
    data[i]:cdiv(old_max - old_min + eps)
    data[i]:add(new_min)
  end
end
return image
