require 'torch'
require 'paths'
require 'image'
image_utils = require 'utils.image'

cifar = {}

cifar.path_dataset = '/home/denton/data/cifar10/cifar-10-batches-t7/'

cifar.coarseSize = 16
cifar.fineSize = 32

-- XXX: Auto contrast normalize in laod script

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
    data = torch.FloatTensor(50000, 3, 32, 32)
    labels = torch.FloatTensor(50000)
    for i = 0,4 do
      local subset = torch.load(cifar.path_dataset .. 'data_batch_' .. (i+1) .. '.t7', 'ascii')
      data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():reshape(10000, 3, 32, 32)
      labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
  else -- load test data
    subset = torch.load(cifar.path_dataset .. 'test_batch.t7', 'ascii')
    data = subset.data:t():reshape(10000, 3, 32, 32):type('torch.FloatTensor')
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
  dataset.data = data -- on cpu
  dataset.labels = labels

  dataset.coarseData = torch.Tensor(N, 3, cifar.coarseSize, cifar.coarseSize)
  dataset.fineData = torch.Tensor(N, 3, cifar.fineSize, cifar.fineSize)
  dataset.diffData = torch.Tensor(N, 3, cifar.fineSize, cifar.fineSize)

  -- Coarse data
  function dataset:makeCoarse()
    for i = 1,N do
      self.coarseData[i][1] = image.scale(self.data[i][1], cifar.coarseSize, cifar.coarseSize)
      self.coarseData[i][2] = image.scale(self.data[i][2], cifar.coarseSize, cifar.coarseSize)
      self.coarseData[i][3] = image.scale(self.data[i][3], cifar.coarseSize, cifar.coarseSize)
    end
  end

  -- Fine data
  function dataset:makeFine()
    for i = 1,N do
      self.fineData[i][1] = image.scale(self.data[i][1], cifar.fineSize, cifar.fineSize)
      self.fineData[i][2] = image.scale(self.data[i][2], cifar.fineSize, cifar.fineSize)
      self.fineData[i][3] = image.scale(self.data[i][3], cifar.fineSize, cifar.fineSize)
    end
  end

  -- Diff (coarse - fine)
  function dataset:makeDiff()
    for i=1,N do
      self.diffData[i][1] = image.scale(self.coarseData[i][1]:float(), cifar.fineSize, cifar.fineSize)
      self.diffData[i][2] = image.scale(self.coarseData[i][2]:float(), cifar.fineSize, cifar.fineSize)
      self.diffData[i][3] = image.scale(self.coarseData[i][3]:float(), cifar.fineSize, cifar.fineSize)

      self.diffData[i][1]:add(-1, self.fineData[i][1])
      self.diffData[i][2]:add(-1, self.fineData[i][2])
      self.diffData[i][3]:add(-1, self.fineData[i][3])
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
       --local input = self.fineData[index]
       local diff = self.diffData[index]
       local cond = self.coarseData[index]
       local class = self.labels[index]
       local fine = self.fineData[index]
       local label = labelvector:zero()
       label[class] = 1
       local example = {diff, class, cond, fine}
       return example
  end})

  return dataset
end
