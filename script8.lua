require 'nn';
if not nn.SpatialConvolutionUpsample then
   dofile('/Users/chinso/code/eyescream_private/layers/SpatialConvolutionUpsample.lua');
end

mo = torch.load('8x14_clean.t7')
print(mo)

mp = nn.ParallelTable()
mp:add(nn.Identity())

mm = nn.Sequential()

l1 = nn.SpatialConvolution(10, 196, 1, 1)
l1.weight:copy(mo[1].weight);
l1.bias:copy(mo[1].bias);
l1.gradWeight = nil
l1.gradBias = nil

mm:add(l1):add(nn.ReLU()):add(nn.View(1, 14, 14):setNumInputDims(3))

mp:add(mm)
mp:add(nn.Identity())

m = nn.Sequential()
m:add(mp):add(nn.JoinTable(2,2))


c1 = nn.SpatialConvolution(mo[2].nInputPlane, mo[2].nOutputPlane, mo[2].kW, mo[2].kH, mo[2].dW, mo[2].dH, mo[2].padding)
c1.weight:copy(mo[2].weight);
c1.bias:copy(mo[2].bias);
c1.gradWeight = nil
c1.gradBias = nil

m:add(c1):add(nn.ReLU())

c2 = nn.SpatialConvolution(mo[3].nInputPlane, mo[3].nOutputPlane, mo[3].kW, mo[3].kH, mo[3].dW, mo[3].dH, mo[3].padding)
c2.weight:copy(mo[3].weight);
c2.bias:copy(mo[3].bias);
c2.gradWeight = nil
c2.gradBias = nil

m:add(c2):add(nn.ReLU())

c3 = nn.SpatialConvolution(mo[4].nInputPlane, mo[4].nOutputPlane, mo[4].kW, mo[4].kH, mo[4].dW, mo[4].dH, mo[4].padding)
c3.weight:copy(mo[4].weight);
c3.bias:copy(mo[4].bias);
c3.gradWeight = nil
c3.gradBias = nil

m:add(c3)

torch.save('8x14_model.t7', m);

one_hot = torch.zeros(1,10,1,1)
one_hot[1][1][1][1] = 1
noise = torch.zeros(1,1,14,14):uniform()
input = torch.randn(1,3,14,14)

out = m:forward({input, noise, one_hot})
print(#out)

require 'image';
image.display(out[1])
