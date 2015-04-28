require 'torch'
require 'nngraph'
require 'cunn'
require 'nnx'
require 'optim'
require 'image'
require 'datasets.coarse_to_fine_cifar10'
require 'pl'
require 'paths'
image_utils = require 'utils.image'
disp = require 'display'
adversarial = require 'train.conditional_adversarial'
debugger = require('fb.debugger')
paths.dofile('../layers/SpatialConvolutionUpsample.lua')


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 10)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -f,--full          (default true)        use the full dataset
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.01)        learning rate, for SGD only
  -b,--batchSize     (default 100)          batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default -1)          on gpu 
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
]]

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end
print(opt)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.coarseSize = 16
opt.fineSize = 32
opt.noiseDim = {1, opt.coarseSize, opt.coarseSize}
classes = {'0','1'}
opt.geometry = {3, opt.fineSize, opt.fineSize}
opt.condDim = {3, opt.coarseSize, opt.coarseSize}

function setWeights(weights, std)
  weights:randn(weights:size())
  weights:mul(std)
end

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  local numhid = 600
  x_I = nn.Identity()()
  x_C = nn.Identity()()
  --i1 = nn.Linear(3*opt.fineSize*opt.fineSize, numhid/2)(nn.Reshape(3*opt.fineSize*opt.fineSize)(x_I))
  --c1 = nn.Linear(3*opt.coarseSize*opt.coarseSize, numhid/2)(nn.Reshape(3*opt.coarseSize*opt.coarseSize)(x_C))
  i1 = nn.SpatialConvolution(3, 128, 5, 5, 2, 2, 2)(x_I)
  c1 = nn.SpatialConvolutionUpsample(3, 128, 5, 5, 1)(x_C)
  h1 = nn.JoinTable(2, 2)({i1, c1})
  --h2 = nn.Linear(numhid, numhid)(nn.ReLU()(h1))
  h2 = nn.SpatialConvolution(256, 256, 5, 5)(nn.ReLU()(h1))
  h3 = nn.Linear(256*12*12, 1)(nn.Dropout()(nn.ReLU()(nn.Reshape(256*12*12)(h2))))
  h4 = nn.Sigmoid()(h3)
  model_D = nn.gModule({x_I, x_C}, {h4})

  --[[
local numhid = 500
local numplanes = 96
  model_D = nn.Sequential()
  model_D:add(nn.SpatialConvolution(3,numplanes,5,5))
  model_D:add(nn.ReLU())
  model_D:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  model_D:add(nn.View(numplanes*13*13))
  model_D:add(nn.Linear(numplanes*13*13, numhid))
  model_D:add(nn.ReLU())
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(numhid,1))
  model_D:add(nn.Sigmoid())
  --]]

  -- Init weights
  setWeights(i1.data.module.weight, 0.005)
  setWeights(c1.data.module.weight, 0.005)
  setWeights(h2.data.module.weight, 0.005)
  setWeights(h3.data.module.weight, 0.005)

  ----------------------------------------------------------------------
  -- define G network to train
  local nplanes = 512 
  model_G = nn.Sequential()
  model_G:add(nn.JoinTable(2, 2))
  model_G:add(nn.SpatialConvolutionUpsample(3+1, nplanes, 1, 1, 1)) -- 3 color channels + conditional
  model_G:add(nn.ReLU())
  model_G:add(nn.SpatialConvolutionUpsample(nplanes, nplanes, 3, 3, 1)) -- 3 color channels + conditional
  model_G:add(nn.Sigmoid())
  model_G:add(nn.SpatialConvolutionUpsample(nplanes, 3, 1, 1, 2)) -- 3 color channels + conditional
  model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))

  -- Init weights
  for i=1,#model_G.modules do
    if model_G.modules[i].weight ~= nil then
      setWeights(model_G.modules[i].weight, 0.05)
      setWeights(model_G.modules[i].bias, 0)
    end
  end
else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

local nparams = 0
for i=1,#model_D.forwardnodes do
  if model_D.forwardnodes[i].data ~= nil and model_D.forwardnodes[i].data.module ~= nil and model_D.forwardnodes[i].data.module.weight ~= nil then
    nparams = nparams + model_D.forwardnodes[i].data.module.weight:nElement()
  end
end
print('\nNumber of free parameters in D: ' .. nparams)
 
local nparams = 0
for i=1,#model_G.modules do
  if model_G.modules[i].weight ~= nil then
    nparams = nparams + model_G.modules[i].weight:nElement()
  end
end
print('Number of free parameters in G: ' .. nparams .. '\n')

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
  ntrain = 40000
  nval = 10000
else
  ntrain = 2000
  nval = 1000
  print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = cifar.loadTrainSet(1, ntrain)
fineMean, fineStd = image_utils.normalize(trainData.fineData)
coarseMean, coarseStd = image_utils.contrastNormalize(trainData.coarseData, -1, 1)

-- create validation set and normalize
valData = cifar.loadTrainSet(ntrain+1, ntrain+nval)
image_utils.normalize(valData.fineData, fineMean, fineStd)
image_utils.contrastNormalize(valData.coarseData, -1, 1)

-- create test set and normalize
testData = cifar.loadTestSet()
--[[
image_utils.normalize(testData.fineData, fineMean, fineStd)
image_utils.normalize(testData.coarseData, coarseMean, coarseStd)
--]]

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()
end

-- Training parameters
sgdState_D = {
  learningRate = opt.learningRate,
  momentum = opt.momentum
}

sgdState_G = {
  learningRate = opt.learningRate,
  momentum = opt.momentum
}

-- training loop
while true do
  -- train/test
  adversarial.train(trainData)
  adversarial.test(valData)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.000004, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.000004, 0.000001)

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    local samples = model_G.modules[#model_G.modules].output
    to_plot = {}
    for i = 1,100 do
      to_plot[i] = samples[i]:float()
    end
--    debugger.enter()
    disp.image(to_plot, {win=opt.window, width=500})
    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
