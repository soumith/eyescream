require 'torch'
require 'paths'
require 'image'
image_utils = require 'utils.image'

cifar = {}

cifar.path_dataset = '/home/denton/data/cifar10/cifar-10-batches-t7/'

cifar.coarseSize = 16
cifar.fineSize = 32

function cifar.init(fineSize, coarseSize)
  cifar.fineSize = fineSize
  cifar.coarseSize = coarseSize
end

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

  dataset.coarseData = torch.Tensor(N, 3, cifar.fineSize, cifar.fineSize)
  dataset.fineData = torch.Tensor(N, 3, cifar.fineSize, cifar.fineSize)
  dataset.diffData = torch.Tensor(N, 3, cifar.fineSize, cifar.fineSize)

  -- Coarse data
  function dataset:makeCoarse()
    for i = 1,N do
      for c=1,3 do
        local tmp = image.scale(self.data[i][c], cifar.coarseSize, cifar.coarseSize)
        self.coarseData[i][c] = image.scale(tmp, cifar.fineSize, cifar.fineSize)
      end
    end
  end

  -- Fine data
  function dataset:makeFine()
    for i = 1,N do
      for c=1,3 do
        self.fineData[i][c] = image.scale(self.data[i][c], cifar.fineSize, cifar.fineSize)
      end
    end
  end

  -- Diff (coarse - fine)
  function dataset:makeDiff()
    for i=1,N do
      for c = 1,3 do
        self.diffData[i][c] = torch.add(self.fineData[i][c], -1, self.coarseData[i][c])
      end
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
