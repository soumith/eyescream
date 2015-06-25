require 'cunn'
require 'cudnn'
require 'fbcunn'
require 'image'
dofile('/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/layers/cudnnSpatialConvolutionUpsample.lua')
dofile('/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/layers/SpatialConvolutionUpsample.lua')

print(nngraph)

model8 = '/home/soumith/local/eyescream/cifar_8x8_classcond_1000_1000_conditional_adversarial.G'
model16 = '/home/soumith/local/eyescream/cifar_coarse_to_fine_8_14_128_128.net'
model32 = '/gfsai-bistro/ai-group/bistro/gpu/soumith/20150523/Eyescream_4_8_16_32_run7.21_53_45.PTeuxS5j/imgslogs/model_4.t7'

m1=torch.load(model8).G
m2 = torch.load(model16).G
--print(m2:forward({torch.ones(1,1,14,14):cuda(), torch.ones(1,3,14,14):cuda()}))
--print(m2:get(2).weight:mean(), m2:get(2).weight:std())
m3 = torch.load(model32).G
m3:evaluate()
m3.modules[#m3.modules] = nil

n = 4

local rands = {}
lrands = {}
rrands = {}

rands[1] = torch.zeros(n,100):uniform(-1,1)
rands[2] = torch.zeros(n,1,14,14):uniform(-1,1)
rands[3] = torch.zeros(n,1,32,32):uniform(-1,1)
rands[4] = torch.zeros(n,1,64,64):uniform(-1,1)
rands[5] = torch.zeros(n,1,128,128):uniform(-1,1)
rands[6] = torch.zeros(n,1,256,256):uniform(-1,1)
rands[7] = torch.zeros(n,1,512,512):uniform(-1,1)


a  = torch.linspace(0,1,32)

cc = torch.CudaTensor(10):zero():view(1,10):repeatTensor(n,1)
for i=1,n do
    cc[i][torch.random(1,10)] = 1
    end

s1 = m1:forward({rands[1]:cuda():view(n,100), cc})
print(#s1)
s11 = image.toDisplayTensor{input=s1, scaleeach=true, nrow=8}
image.save('/home/soumith/public_html/1.png', s11)

local s2_i = torch.zeros(n, 3, 14, 14)
for i=1,n do
    s2_i[i]:copy(image.scale(s1[i]:float(), 14, 14))
    end
s2_i = s2_i:cuda()
s12 = image.toDisplayTensor{input=s2_i, scaleeach=true, nrow=8}
image.save('/home/soumith/public_html/11.png', s12)

s2 = m2:forward({rands[2]:cuda(), s2_i:view(n,3,14,14)})
print(#s2)
s2_i = torch.add(s2_i, s2) -- s2_i:add(s2)
s22 = image.toDisplayTensor{input=s2_i, scaleeach=true, nrow=8}
s2_o = image.toDisplayTensor{input=s2, scaleeach=true, nrow=8}
image.save('/home/soumith/public_html/2.png', s22)
image.save('/home/soumith/public_html/2o.png', s2_o)

local s3_i = torch.zeros(n, 3, 32, 32)
for i=1,n do
    s3_i[i]:copy(image.scale(s2_i[i]:float(), 32, 32))
    end
s3_i = s3_i:cuda()
s3 = m3:forward({rands[3]:cuda(), s3_i:view(n,3,32,32)})

print(#s3)
s3_i:add(s3)
s33 = image.toDisplayTensor{input=s3_i, scaleeach=true, nrow=8}
image.save('/home/soumith/public_html/3.png', s33)

local s4_i = torch.zeros(n, 3, 64, 64)
for i=1,n do
    s4_i[i]:copy(image.scale(s3_i[i]:float(), 64, 64))
    end
s4_i = s4_i:cuda()
s4 = m3:forward({rands[4]:cuda(), s4_i:view(n,3,64,64)})

print(#s4)
s4_i:add(s4)
s44 = image.toDisplayTensor{input=s4_i, scaleeach=true, nrow=8}
image.save('/home/soumith/public_html/4.png', s44)

local s5_i = torch.zeros(n, 3, 128, 128)
for i=1,n do
    s5_i[i]:copy(image.scale(s4_i[i]:float(), 128, 128))
    end
s5_i = s5_i:cuda()
s5 = m3:forward({rands[5]:cuda(), s5_i:view(n,3,128,128)})

print(#s5)
s5_i:add(s5)
s54 = image.toDisplayTensor{input=s5_i, scaleeach=true, nrow=8}
image.save('/home/soumith/public_html/5.png', s54)

local s6_i = torch.zeros(n, 3, 256, 256)
for i=1,n do
    s6_i[i]:copy(image.scale(s5_i[i]:float(), 256, 256))
    end
s6_i = s6_i:cuda()
s6 = m3:forward({rands[6]:cuda(), s6_i:view(n,3,256,256)})

print(#s6)
s6_i:add(s6)
s65 = image.toDisplayTensor{input=s6_i, scaleeach=true, nrow=8}
image.save('/home/soumith/public_html/6.png', s65)
