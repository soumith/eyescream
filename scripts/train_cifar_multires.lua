require 'torch'
require 'nngraph'
require 'cunn'
require 'nnx'
require 'optim'
require 'image'
require 'datasets.multires_cifar10'
require 'pl'
require 'paths'
image_utils = require 'utils.image'
disp = require 'display'
adversarial = require 'train.multires_conditional_adversarial'
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

opt.numres = 2
opt.coarseSize = {8, 16}
opt.fineSize = {16, 32}
opt.noiseDim = { {1, opt.fineSize[1], opt.fineSize[1]}, 
                 {1, opt.fineSize[2], opt.fineSize[2]}}
classes = {'0','1'}
opt.geometry = { {3, opt.fineSize[1], opt.fineSize[1]}, 
                 {3, opt.fineSize[2], opt.fineSize[2]}}
opt.condDim = {{3, opt.fineSize[1], opt.fineSize[1]}, 
               {3, opt.fineSize[2], opt.fineSize[2]}}

function setWeights(weights, std)
  weights:randn(weights:size())
  weights:mul(std)
end


if opt.network == '' then
  ----------------------------------------------------------------------
  -- define D network to train
  local nplanes = 64 
  model_D = {}
  for r = 1,opt.numres do
    input_sz = opt.geometry[r][1] * opt.geometry[r][2] * opt.geometry[r][3]
    model_D[r] = nn.Sequential()
    model_D[r]:add(nn.CAddTable())
    model_D[r]:add(nn.SpatialConvolution(3, nplanes, 5, 5)) --28 x 28
    model_D[r]:add(nn.ReLU())
    model_D[r]:add(nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2))
    local sz =math.floor( ( (opt.fineSize[r] - 5 + 1) - 5) / 2 + 1)
    model_D[r]:add(nn.Reshape(nplanes*sz*sz))
    model_D[r]:add(nn.ReLU())
    model_D[r]:add(nn.Linear(nplanes*sz*sz, 1))
    model_D[r]:add(nn.Sigmoid())
  end

  ----------------------------------------------------------------------
  -- define G network to train
  
  local nplanes = 64 
  model_G = {}
  for r = 1,opt.numres do
    model_G[r] = nn.Sequential()
    model_G[r]:add(nn.JoinTable(2, 2))
    model_G[r]:add(nn.SpatialConvolutionUpsample(3+1, nplanes, 7, 7, 1)) -- 3 color channels + conditional
    model_G[r]:add(nn.ReLU())
    model_G[r]:add(nn.SpatialConvolutionUpsample(nplanes, nplanes, 7, 7, 1)) -- 3 color channels + conditional
    model_G[r]:add(nn.ReLU())
    model_G[r]:add(nn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 1)) -- 3 color channels + conditional
    model_G[r]:add(nn.View(opt.geometry[r][1], opt.geometry[r][2], opt.geometry[r][3]))
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
parameters_D = {}
parameters_G = {}
gradParameters_D = {}
gradParameters_G = {}
for r = 1,opt.numres do
  parameters_D[r],gradParameters_D[r] = model_D[r]:getParameters()
  parameters_G[r],gradParameters_G[r] = model_G[r]:getParameters()
end

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

for r = 1,opt.numres do
  local nparams = 0
  for i=1,#model_D[r].modules do
    if model_D[r].modules[i].weight ~= nil then
      nparams = nparams + model_D[r].modules[i].weight:nElement()
    end
  end
  print('\nNumber of free parameters in D' .. r .. ': ' .. nparams)
   
  local nparams = 0
  for i=1,#model_G[r].modules do
    if model_G[r].modules[i].weight ~= nil then
      nparams = nparams + model_G[r].modules[i].weight:nElement()
    end
  end
  print('Number of free parameters in G' .. r .. ': ' .. nparams .. '\n')
end

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
  ntrain = 45000
  nval = 5000
else
  ntrain = 2000
  nval = 1000
  print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = cifar.loadTrainSet(1, ntrain)
--image_utils.contrastNormalize(trainData.data, -1, 1)
mean, std = image_utils.normalize(trainData.data)
local defaultType = torch.getdefaulttensortype()
torch.setdefaulttensortype('torch.FloatTensor')
trainData:makeFine()
trainData:makeCoarse()
trainData:makeDiff()
torch.setdefaulttensortype(defaultType)

-- create validation set and normalize
valData = cifar.loadTrainSet(ntrain+1, ntrain+nval)
--image_utils.contrastNormalize(valData.data, -1, 1)
image_utils.normalize(valData.data, mean, std)
torch.setdefaulttensortype('torch.FloatTensor')
valData:makeFine()
valData:makeCoarse()
valData:makeDiff()
torch.setdefaulttensortype(defaultType)

-- create test set and normalize
--[[
testData = cifar.loadTestSet()
image_utils.normalize(testData.fineData, fineMean, fineStd)
image_utils.normalize(testData.coarseData, coarseMean, coarseStd)
--]]

-- this matrix records the current confusion across classes
confusion = {}
for r = 1,opt.numres do
  confusion[r] = optim.ConfusionMatrix(classes)
end

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  for r = 1,opt.numres do
    model_D[r]:cuda()
    model_G[r]:cuda()
  end
end

-- Training parameters
sgdState_D = {}
for r = 1,opt.numres do
  sgdState_D[r] = {
    learningRate = opt.learningRate,
    momentum = opt.momentum
  }
end

sgdState_G = {}
for r = 1,opt.numres do
  sgdState_G[r] = {
    learningRate = opt.learningRate,
    momentum = opt.momentum
  }
end

function scaleColor(img, w)
  local scaled = torch.FloatTensor(3, w, w)
  for c = 1,3 do
    scaled[c] = image.scale(img[c]:float(), w, w)
  end
  return scaled
end

-- Get examples to plot
function getSamples(dataset, N)
  local N = N or 8 
  local inputs = {}
  local noise_inputs = {}
  local cond_inputs = {}
  local gt = {}
  local generated_samples = {}
  local diff = {}
  for i = 1,opt.numres do
    inputs[i] = torch.Tensor(N, opt.geometry[i][1], opt.geometry[i][2], opt.geometry[i][3])
    noise_inputs[i] = torch.Tensor(N, opt.noiseDim[i][1], opt.noiseDim[i][2], opt.noiseDim[i][3])
    cond_inputs[i] = torch.Tensor(N, opt.condDim[i][1], opt.condDim[i][2], opt.condDim[i][3])
    gt[i] = torch.Tensor(N, opt.geometry[i][1], opt.geometry[i][2], opt.geometry[i][3])
    generated_samples[i] = torch.Tensor(N, opt.geometry[i][1], opt.geometry[i][2], opt.geometry[i][3])
    diff[i] = torch.Tensor(N, opt.geometry[i][1], opt.geometry[i][2], opt.geometry[i][3])
  end

  local rand = torch.Tensor(N)
  for i = 1,N do
    rand[i] = math.random(dataset:size())
  end

  for r = 1,opt.numres do
    noise_inputs[r]:uniform(-1, 1)
    for i = 1,N do
      local sample = dataset[r][rand[i]]
      if r == 1 then
        cond_inputs[r][i] = sample[3]:clone()
      else
        cond_inputs[r][i] = scaleColor(generated_samples[r-1][i], opt.condDim[r][2])
      end
      gt[r][i] = dataset[r].fineData[rand[i]]
    end
    local samples = model_G[r]:forward({noise_inputs[r], cond_inputs[r]})
    diff[r]:copy(samples)
    generated_samples[r] = torch.add(samples, cond_inputs[r])
  end
    
  local to_plot = {}
  for i=1,N do
    to_plot[#to_plot+1] = gt[2][i]
    to_plot[#to_plot+1] = generated_samples[2][i]
    to_plot[#to_plot+1] = scaleColor(diff[2][i], 32) 
    to_plot[#to_plot+1] = cond_inputs[2][i]
    to_plot[#to_plot+1] = scaleColor(diff[1][i], 32) 
    to_plot[#to_plot+1] = scaleColor(cond_inputs[1][i], 32) 
  end

  return to_plot
end

--[[
local to_plot = getSamples(valData, 6) 
torch.setdefaulttensortype('torch.FloatTensor')
disp.image(to_plot, {win=opt.window, width=600})
debugger.enter()
--]]

-- training loop
while true do
  -- train/test
  adversarial.train(trainData)
  adversarial.test(valData)

  for r = 1,opt.numres do
    sgdState_D[r].momentum = math.min(sgdState_D[r].momentum + 0.0008, 0.7)
    sgdState_D[r].learningRate = math.max(sgdState_D[r].learningRate / 1.000004, 0.000001)
    sgdState_G[r].momentum = math.min(sgdState_G[r].momentum + 0.0008, 0.7)
    sgdState_G[r].learningRate = math.max(sgdState_G[r].learningRate / 1.000004, 0.000001)
  end

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    local to_plot = getSamples(valData, 24) 
    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    disp.image(to_plot, {win=opt.window, width=600})
    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
