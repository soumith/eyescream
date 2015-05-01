require 'torch'
require 'paths'

mnist = {}

mnist.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
mnist.path_dataset = os.getenv('HOME') .. '/code/mnist/mnist.t7/'
mnist.path_trainset = mnist.path_dataset ..  'train_32x32.t7'
mnist.path_testset = mnist.path_dataset .. 'test_32x32.t7'

function mnist.download()
   if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
      local remote = mnist.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.loadTrainSet(start, stop)
   return mnist.loadDataset(mnist.path_trainset, start, stop)
end

function mnist.loadTestSet()
   return mnist.loadDataset(mnist.path_testset)
end

function mnist.loadDataset(fileName, start, stop)
   --mnist.download()
   local f = torch.load(fileName, 'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   local labels = f.labels

   local nExample = f.data:size(1)
   local start = start or 1
   local stop = stop or nExample
   if stop > nExample then
      stop = nExample
   end
   local data = data[{{start,stop},{},{3,30},{3,30}}]
   local labels = labels[{{start, stop}}]

   print('<mnist> loaded ' .. stop - start + 1 .. ' examples')

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
           local example = {input, class, label} -- class cond vector is label vector
                                       return example
   end})

   return dataset
end
