require 'torch'
require 'nngraph'
require 'cunn'
require 'nnx'
require 'optim'
require 'image'
require 'datasets.cifar10'
require 'pl'
require 'paths'
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


opt.condDim = 10
classes = {'0','1'}
opt.geometry = {3, 32, 32}

function setWeights(weights, std)
  weights:randn(weights:size())
  weights:mul(std)
end

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  local numhid = 1600
  x_I = nn.Identity()()
  x_C = nn.Identity()()
  c1 = nn.JoinTable(2, 2)({nn.Reshape(input_sz)(x_I), x_C})
  h1 = nn.Linear(input_sz + opt.condDim, numhid)(c1)
  h2 = nn.Linear(numhid, numhid)(nn.Dropout()(nn.ReLU()(h1)))
  h3 = nn.Linear(numhid, 1)(nn.Dropout()(nn.ReLU()(h2)))
  out = nn.Sigmoid()(h3)
  model_D = nn.gModule({x_I, x_C}, {out})

  -- Init weights
  setWeights(h1.data.module.weight, 0.005)
  setWeights(h2.data.module.weight, 0.005)
  setWeights(h3.data.module.weight, 0.005)

  ----------------------------------------------------------------------
  -- define G network to train
  local numhid = 8000
--[[
  model_G = nn.Sequential()
  model_G:add(nn.Linear(opt.noiseDim, numhid))
  model_G:add(nn.ReLU())
  model_G:add(nn.Linear(numhid, numhid))
  model_G:add(nn.Sigmoid())
  model_G:add(nn.Linear(numhid, input_sz))
  model_G:add(nn.Reshape(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
 --]] 
 --
  local nplanes = 180 
  local w = 8
  local numhid = nplanes*w*w
  model_G = nn.Sequential()
  model_G:add(nn.JoinTable(2, 2))
  model_G:add(nn.Linear(opt.noiseDim + opt.condDim, 8000))
  model_G:add(nn.ReLU())
  model_G:add(nn.Linear(8000, nplanes*w*w))
  model_G:add(nn.Sigmoid())
  model_G:add(nn.View(nplanes, w, w))
  model_G:add(nn.SpatialConvolutionUpsample(nplanes, 3, 7, 7, 4))
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

--debugger.enter()
-- create training set and normalize
trainData = cifar.loadTrainSet(1, ntrain)
mean, std = trainData:normalize()

-- create validation set and normalize
valData = cifar.loadTrainSet(ntrain+1, ntrain+nval)
mean, std = valData:normalize()

-- create test set and normalize
--[[
testData = cifar.loadTestSet()
testData:normalize(mean, std)
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

-- Get examples to plot
function getSamples(dataset, N)
  local N = N or opt.batchSize
  local noise_inputs = torch.Tensor(N, opt.noiseDim)
  local cond_inputs = torch.zeros(N, 10)

  -- Generate samples 
  noise_inputs:uniform(-1, 1)
  local nc = math.floor(math.sqrt(N))
  local class = math.random(10) 
  for n = 1,N do
    if n % nc == 0 then
      class = math.random(10) 
    end
    local rand = math.random(dataset:size())
    local sample = dataset[rand]
    cond_inputs[n][class] = 1 
  end
  local samples = model_G:forward({noise_inputs, cond_inputs})

  local to_plot = {}
  for i=1,N do
    to_plot[i] = samples[i]:float()
  end

  return to_plot
end

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

    local to_plot = getSamples(valData, 64)

    torch.setdefaulttensortype('torch.FloatTensor')
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    disp.image(to_plot, {labels = labels, win=opt.window, width=500})
    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
