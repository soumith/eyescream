require 'torch'
require 'nn'
require 'cunn'
require 'nnx'
require 'optim'
require 'image'
require 'datasets.mnist'
require 'pl'
require 'paths'
disp = require 'display'
adversarial = require 'train.conditional_adversarial'
debugger = require('fb.debugger')


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
  -w, --window       (default 2)           windsow id of sample image
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
opt.geometry = {1, 28, 28}

function setWeights(weights, std)
  weights:randn(weights:size())
  weights:mul(std)
end

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
----------------------------------------------------------------------
-- define D network to train
  model_D = nn.Sequential()
  model_D:add(nn.JoinTable(2, 2))
  model_D:add(nn.Linear(input_sz + 10, 240))
  model_D:add(nn.ReLU())
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(240, 240))
  model_D:add(nn.ReLU())
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(240,1))
  model_D:add(nn.Sigmoid())

  -- Init weights
  setWeights(model_D.modules[2].weight, 0.005)
  setWeights(model_D.modules[5].weight, 0.005)
  setWeights(model_D.modules[8].weight, 0.005)
  setWeights(model_D.modules[2].bias, 0)
  setWeights(model_D.modules[5].bias, 0)
  setWeights(model_D.modules[8].bias, 0)

  ----------------------------------------------------------------------
  -- define G network to train
  model_G = nn.Sequential()
  model_G:add(nn.JoinTable(2, 2))
  model_G:add(nn.Linear(opt.noiseDim + 10, 1200))
  model_G:add(nn.ReLU())
  model_G:add(nn.Linear(1200, 1200))
  model_G:add(nn.ReLU())
  model_G:add(nn.Linear(1200, input_sz))
  model_G:add(nn.Sigmoid())

  -- Init weights
  setWeights(model_G.modules[2].weight, 0.05)
  setWeights(model_G.modules[4].weight, 0.05)
  setWeights(model_G.modules[6].weight, 0.05)
  setWeights(model_G.modules[2].bias, 0)
  setWeights(model_G.modules[4].bias, 0)
  setWeights(model_G.modules[6].bias, 0)
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


----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
  ntrain = 55000
  nval = 5000
else
  ntrain = 2000
  nval = 1000
  print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(1, ntrain)
mean, std = trainData:normalize()
trainData:contrastNormalize(0, 1)

-- create validation set and normalize
valData = mnist.loadTrainSet(ntrain+1, ntrain+nval)
mean, std = valData:normalize()
valData:contrastNormalize(0, 1)

-- create test set and normalize
testData = mnist.loadTestSet()
testData:normalize(mean, std)
testData:contrastNormalize(0, 1)

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
  local cond_inputs = adversarial.test(valData)

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0002, 0.7)
  sgdState_D.learningRate = math.max(sgdState_D.learningRate / 1.000004, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0002, 0.7)
  sgdState_G.learningRate = math.max(sgdState_G.learningRate / 1.000004, 0.000001)

  -- plot errors
  if opt.plot then
    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    local samples = model_G.modules[#model_G.modules].output
    to_plot = {}
    labels = {}
    for i = 1,100 do
      to_plot[i] = samples[i]:float():reshape(opt.geometry[1], opt.geometry[2], opt.geometry[3])
      for j = 1,10 do
        if cond_inputs[i][j] == 1 then
          labels[i] = tostring(j-1)
          break
        end
      end
    end
--    debugger.enter()
    disp.image(to_plot, {win=opt.window, width=500, labels=labels})
    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
