require 'torch'
require 'paths'
require 'image'

cifar = {}

cifar.path_dataset = '/home/denton/data/cifar10/cifar-10-batches-t7/'

cifar.coarseSize = 16
cifar.fineSize = 32

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
  local N = stop - start + 1
  print('<cifar10> loaded ' .. N .. ' examples') 

  local dataset = {}

  -- Coarse data
  dataset.coarseData = torch.Tensor(N, 3, cifar.coarseSize, cifar.coarseSize)
  for i = 1,N do
    dataset.coarseData[i][1] = image.scale(data[i][1]:float(), cifar.coarseSize, cifar.coarseSize):type(torch.getdefaulttensortype())
    dataset.coarseData[i][2] = image.scale(data[i][2]:float(), cifar.coarseSize, cifar.coarseSize):type(torch.getdefaulttensortype())
    dataset.coarseData[i][3] = image.scale(data[i][3]:float(), cifar.coarseSize, cifar.coarseSize):type(torch.getdefaulttensortype())
  end

  -- Fine data
  if cifar.fineSize == 32 then
    dataset.fineData = data 
  else
    dataset.fineData = torch.Tensor(N, 3, cifar.fineSize, cifar.fineSize)
    for i = 1,N do
      dataset.fineData[i][1] = image.scale(data[i][1]:float(), cifar.fineSize, cifar.fineSize):type(torch.getdefaulttensortype())
      dataset.fineData[i][2] = image.scale(data[i][2]:float(), cifar.fineSize, cifar.fineSize):type(torch.getdefaulttensortype())
      dataset.fineData[i][3] = image.scale(data[i][3]:float(), cifar.fineSize, cifar.fineSize):type(torch.getdefaulttensortype())
    end
  end

  -- Labels
  dataset.labels = labels

  function dataset:size()
    return stop - start + 1 
  end

  function dataset:numClasses()
    return 10
  end

  local labelvector = torch.zeros(10)

  setmetatable(dataset, {__index = function(self, index)
       local input = self.fineData[index]
       local cond = self.coarseData[index]
       local class = self.labels[index]
       local label = labelvector:zero()
       label[class] = 1
       local example = {input, class, cond}
       return example
  end})

  return dataset
end
