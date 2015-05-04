require 'torch'
require 'paths'
require 'image'
image_utils = require 'utils.image'

cifar = {}

cifar.path_dataset = '/home/denton/data/cifar10/cifar-10-batches-t7/'

cifar.sizes = {8, 16, 32}

function cifar.init(sizes)
  cifar.sizes = sizes
end

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

  local datasets = {}
  datasets.data = data -- on cpu
  datasets.labels = labels

  for d = 1,#cifar.sizes-1 do
    local dataset = {}
    dataset.coarseSize = cifar.sizes[d]
    dataset.fineSize = cifar.sizes[d+1]

    dataset.fineData = torch.Tensor(N, 3, dataset.fineSize, dataset.fineSize)
    dataset.coarseData = torch.Tensor(N, 3, dataset.fineSize, dataset.fineSize)
    dataset.diffData = torch.Tensor(N, 3, dataset.fineSize, dataset.fineSize)

    datasets[d] = dataset
  end

  -- Coarse data
  function datasets:makeCoarse()
    for s = 1,#cifar.sizes-1 do
      local coarseSize = self[s].coarseSize
      local fineSize = self[s].fineSize
      for i = 1,N do
        for c=1,3 do
          local tmp = image.scale(self.data[i][c], coarseSize, coarseSize)
          self[s].coarseData[i][c] = image.scale(tmp, fineSize, fineSize)
        end
      end
    end
  end

  -- Fine data
  function datasets:makeFine()
    for s = 1,#cifar.sizes-1 do
      local fineSize = self[s].fineSize
      for i = 1,N do
        for c=1,3 do
          self[s].fineData[i][c] = image.scale(self.data[i][c], fineSize, fineSize)
        end
      end
    end
  end

  -- Diff (coarse - fine)
  function datasets:makeDiff()
    for s = 1,#cifar.sizes-1 do
      for i=1,N do
        self[s].diffData[i] = torch.add(self[s].fineData[i], -1, self[s].coarseData[i])
      end
    end
  end

  function datasets:size()
    return stop - start + 1 
  end

  for s = 1,#cifar.sizes-1 do
    setmetatable(datasets[s], {__index = function(self, index)
         --local input = self.fineData[index]
         local diff = self.diffData[index]
         local cond = self.coarseData[index]
         local fine = self.fineData[index]
         local example = {diff, nil, cond, fine}
         return example
    end})
  end
  return datasets
end
