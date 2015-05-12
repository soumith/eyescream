require 'torch'
require 'cunn'
require 'optim'
require 'image'
require 'paths'
disp = require 'display'
paths.dofile('layers/SpatialConvolutionUpsample.lua')

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 10)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.01)        learning rate, for SGD only
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 1)          on gpu
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --nDonkeys         (default 2)           number of data loading threads
  --cache            (default "cache")     folder to cache metadata
  --data             (default "/home/awesomebox/imagenet-256/256") folder with imagenet data
  --epochSize        (default 1000)        number of samples per epoch
]]

print(opt)

-- fix seed
opt.manualSeed = 1
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
cutorch.setDevice(opt.gpu)
torch.setdefaulttensortype('torch.FloatTensor')
opt.coarseSize = 16
opt.fineSize = 32
opt.loadSize = 48
opt.noiseDim = {1, opt.fineSize, opt.fineSize}
classes = {'0','1'}
opt.geometry = {3, opt.fineSize, opt.fineSize}
opt.condDim = {3, opt.fineSize, opt.fineSize}

paths.dofile('model.lua')
paths.dofile('data.lua')
adversarial = paths.dofile('conditional_adversarial.lua')

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Training parameters
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum
}

sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum
}

local function train()
   confusion:zero()
   model_D:training()
   model_G:training()
   for i=1,opt.epochSize do
      xlua.progress(i, opt.epochSize)
      donkeys:addjob(
         function()
            return makeData(trainLoader:sample(opt.batchSize))
         end,
         adversarial.train)
   end
   donkeys:synchronize()
   cutorch.synchronize()
   print(confusion)
end

local function test()
   confusion:zero()
   model_D:evaluate()
   model_G:evaluate()
   for i=1,nTest/opt.batchSize do -- nTest is set in data.lua
      xlua.progress(i, math.floor(nTest/opt.batchSize))
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(function() return makeData(testLoader:get(indexStart, indexEnd)) end,
         adversarial.test)
   end
   donkeys:synchronize()
   cutorch.synchronize()
   print(confusion)
end

local function plot()
end

epoch = 1
while true do
   train()
   test()
   sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
   sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.000004, 0.000001)
   sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
   sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.000004, 0.000001)

   plot()
   epoch = epoch + 1
end
