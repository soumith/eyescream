require 'nn'
dofile('SpatialConvolutionUpsample.lua')

m = nn.SpatialConvolutionUpsample(3,16,5,5)
inp = torch.randn(3, 32, 32)

o=m:forward(inp)
print(#o)

m:backward(inp, torch.randn(16, 64, 64))
