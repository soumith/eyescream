  ----------------------------------------------------------------------
  -- define D network to train
  local nplanes = 128 
  model_D = nn.Sequential()
  model_D:add(nn.CAddTable())
  model_D:add(nn.SpatialConvolution(3, nplanes, 5, 5)) --28 x 28
  model_D:add(nn.ReLU())
  model_D:add(nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2))
  model_D:add(nn.Reshape(nplanes*12*12))
  model_D:add(nn.ReLU())
  model_D:add(nn.Linear(nplanes*12*12, 1))
  model_D:add(nn.Sigmoid())
  ----------------------------------------------------------------------
  -- define G network to train  
  local nplanes = 128 
  model_G = nn.Sequential()
  model_G:add(nn.JoinTable(2, 2))
  model_G:add(nn.SpatialConvolutionUpsample(3+1, nplanes, 7, 7, 1)) -- 3 color channels + conditional
  model_G:add(nn.ReLU())
  model_G:add(nn.SpatialConvolutionUpsample(nplanes, nplanes, 7, 7, 1)) -- 3 color channels + conditional
  model_G:add(nn.ReLU())
  model_G:add(nn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 1)) -- 3 color channels + conditional
  model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
