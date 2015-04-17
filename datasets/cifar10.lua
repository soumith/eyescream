require 'torch'
require 'paths'

cifar = {}

cifar.path_dataset = '/home/denton/data/cifar10/cifar-10-batches-t7/'

function cifar.loadTrainSet(start, stop)
   return cifar.loadDataset(true, start, stop)
end

function cifar.loadTestSet()
   return cifar.loadDataset(false)
end

function cifar.loadDataset(isTrain, start, stop)
  local data
  local labels
  local defaultType = torch.getdefaulttensortype()
  if isTrain then -- load train data
    data = torch.Tensor(50000, 3, 32, 32)
    labels = torch.Tensor(50000)
    for i = 0,4 do
      local subset = torch.load(cifar.path_dataset .. 'data_batch_' .. (i+1) .. '.t7', 'ascii')
      data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():reshape(10000, 3, 32, 32):type(defaultType)
      labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels:type(defaultType)
    end
  else -- load test data
    subset = torch.load(cifar.path_dataset .. 'test_batch.t7', 'ascii')
    data = subset.data:t():reshape(10000, 3, 32, 32):type(defaultType)
    labels = subset.labels:t():type(defaultType)
  end
   
  local start = start or 1
  local stop = stop or data:size(1)

  -- select chunk
  data = data[{ {start, stop} }]
  labels = labels[{ {start, stop} }]
  labels:add(1) -- becasue indexing is 1-based
  print('<cifar10> loaded ' .. stop - start + 1 .. ' examples') 

  local dataset = {}
  dataset.data = data
  dataset.labels = labels

   function dataset:normalize(mean_, std_)
      local mean = mean or data:mean(1)
      local std = std_ or data:std(1, true)
      local eps = 1e-7
      for i=1,data:size(1) do
         data[i]:add(-1, mean)
         data[i]:cdiv(std + eps)
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

  function dataset:contrastNormalize(new_min, new_max)
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

   function dataset:size()
      return stop - start + 1 
   end

   function dataset:numClasses()
     return 10
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
           local input = self.data[index]
           local class = self.labels[index]
           local label = labelvector:zero()
           label[class] = 1
           local example = {input, class, label}
                                       return example
   end})

   return dataset
end
