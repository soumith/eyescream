require 'cunn'
require 'cudnn'
require 'fbcunn'
require 'image'
dofile('/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/layers/cudnnSpatialConvolutionUpsample.lua')
dofile('/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/layers/SpatialConvolutionUpsample.lua')
torch.manualSeed(1)

do64 = true
n = 128

tt = 'restaurant'
print(tt)
classes = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
          'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}

if tt == 'bedroom' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_19_03.zFh5rsQl/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_20_42.Z2dYiZA5/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_16_16.im3i84Al/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image1_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge1_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_bedroom_.t7')
elseif tt == 'bridge' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_17_35.NJ5DTv6e/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_21_15.hj25UP8C/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_18_12.r0AqX1dn/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image2_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge2_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_bridge_.t7')
elseif tt == 'church_outdoor' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_21_22.uYrJCQZF/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_20_55.I01XSKlS/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_20_02.277Gt90q/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image3_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge3_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_church_outdoor_.t7')
elseif tt == 'classroom' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_18_05.AeNREkk9/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_17_09.rHezCV6Y/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_19_17.GF9EFdV3/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image4_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge4_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_classroom_.t7')
elseif tt == 'conference_room' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_19_30.NuSoOTo8/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_16_29.ru6f0LRb/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_21_37.rsX4yTAG/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image5_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge5_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_conference_room_.t7')
elseif tt == 'dining_room' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_17_21.tlO8eS0A/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_20_49.DSYyuYmV/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_21_01.AkqHwvJo/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image6_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge6_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_dining_room_.t7')
elseif tt == 'kitchen' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_16_36.ZPxPdKik/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_19_23.0sVoJxN7/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_18_25.7EF0OQZa/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image7_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge7_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_kitchen_.t7')
elseif tt == 'living_room' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_17_02.8QND9PYv/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_19_10.YLLAbYs3/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_18_57.dDyBFQ01/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image8_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge8_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_living_room_.t7')
elseif tt == 'restaurant' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_17_28.RKYDfHvU/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_19_43.wxW0ZBY2/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_17_15.pVb19CFM/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image9_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge9_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_restaurant_.t7')
elseif tt == 'tower' then
   model4 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/cifar_4x4_100_100_adversarial.net'
   model8 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_19_49.uJLyF8vQ/imgslogs/csize_4_model_4t8_dset_lsun.t7'
   model16 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_17_52.bXh8WFpA/imgslogs/csize_8_model_large_dset_lsun.t7'
   model32 = '/gfsai-bistro/ai-group/bistro/gpu/aszlam/20150603/lsun_per_category4.22_21_08.t2OjUO7T/imgslogs/csize_16_model_large_dset_lsun.t7'
   model64 = '/home/soumith/local/fbcode2/deeplearning/experimental/soumith/eyescream/32_64.net'
   imgdir = '/gfsai/ai-group/users/aszlam/lsun_test_images/image10_num_'
   imgldir = '/gfsai/ai-group/users/aszlam/lsun_test_images/imagelarge10_num_'
   meanstd = torch.load('/gfsai-cached/ai-group/datasets/lsun/meanstdCache_tower_.t7')
end
mean = meanstd.mean
std = meanstd.std

m0=torch.load(model4).G
m1=torch.load(model8).G
m2 = torch.load(model16).G
print(#m2:forward({torch.ones(1,1,16,16):cuda(), torch.ones(1,3,16,16):cuda()}))
print(m2:get(2).weight:mean(), m2:get(2).weight:std())
m3 = torch.load(model32).G
m3:evaluate()
m3.modules[#m3.modules] = nil

m4 = torch.load(model64).G
m4:evaluate()
m4.modules[#m4.modules] = nil

sets = {}

for counter=1,10 do
   local rands = {}
   rands[0] = torch.zeros(n,100):uniform(-1,1)
   rands[1] = torch.zeros(n,1, 8, 8):uniform(-1,1) -- :mul(counter)
   rands[2] = torch.zeros(n,1,16,16):uniform(-1,1) -- :mul(counter)
   rands[3] = torch.zeros(n,1,32,32):uniform(-1,1) -- :mul(counter)
   rands[4] = torch.zeros(n,1,64,64):uniform(-1,1) -- :mul(counter)
   rands[5] = torch.zeros(n,1,128,128):uniform(-1,1) -- :mul(counter)

   local imgput = torch.zeros(n, 3, 8, 8):float()
   local loadSize   = {3, 12}
   local sampleSize = {3, 8}
   local function loadImage(blob)
      local input = image.load(blob, 3, 'float')
      -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
      local iW = input:size(3)
      local iH = input:size(2)
      if iW < iH then
         input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
      else
         input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
      end
      return input
   end
   for i=1,n do
      local input = loadImage(imgdir .. i .. '.png')
      local oH = sampleSize[2];
      local oW = sampleSize[2];
      local iW = input:size(3)
      local iH = input:size(2)
      local w1 = math.ceil((iW-oW)/2)
      local h1 = math.ceil((iH-oH)/2)
      local out = image.crop(input, w1, h1, w1+oW, h1+oW) -- center patch
      -- mean/std
      for i=1,3 do -- channels
         out[{{i},{},{}}]:add(-mean[i])
         out[{{i},{},{}}]:div(std[i])
      end
      local tmp = image.scale(out, 4, 4)
      image.scale(imgput[i], tmp, 'bilinear')
   end
   s00 = image.toDisplayTensor{input=imgput, scaleeach=true, nrow=8}
   image.save('/home/soumith/public_html/0.png', s00)
   s0_i = imgput

   for j=1,n do
      sets[j] = sets[j] or {}
      local inputl = image.load(imgldir .. j .. '.png')
      for i=1,3 do -- channels
         inputl[{{i},{},{}}]:mul(std[i])
         inputl[{{i},{},{}}]:add(mean[i])
      end
      -- print(#inputl)
      -- print(imgldir .. j .. '.png')
      sets[j][1] = sets[j][1] or inputl
      sets[j][2] = sets[j][2] or s0_i[j]
   end

   s1 = m1:forward({rands[1], s0_i})
   print(#s1)

   s1 = torch.add(s1:float(), s0_i:float())
   s11 = image.toDisplayTensor{input=s1, scaleeach=true, nrow=8}
   image.save('/home/soumith/public_html/1.png', s11)

   local s2_i = torch.zeros(n, 3, 16, 16)
   for i=1,n do
      s2_i[i]:copy(image.scale(s1[i]:float(), 16, 16))
   end
   s2_i = s2_i:cuda()
   s12 = image.toDisplayTensor{input=s2_i, scaleeach=true, nrow=8}
   image.save('/home/soumith/public_html/11.png', s12)

   s2 = m2:forward({rands[2]:cuda(), s2_i:view(n,3,16,16)})
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

   if do64 then
      local s4_i = torch.zeros(n, 3, 64, 64)
      for i=1,n do
         s4_i[i]:copy(image.scale(s3_i[i]:float(), 64, 64))
      end
      s4_i = s4_i:cuda()
      s4 = m4:forward({rands[4]:cuda(), s4_i:view(n,3,64,64)})

      print(#s4)
      s4_i:add(s4)
      s44 = image.toDisplayTensor{input=s4_i, scaleeach=true, nrow=8}
      image.save('/home/soumith/public_html/4.png', s44)

      for j=1,n do
         sets[j][2+counter] = s4_i[j]
      end
   end
end

print(sets)

finsets = {}
for i=1,#sets do
   sets[i][1] = image.scale(sets[i][1], 64, 64)
   sets[i][2] = image.scale(sets[i][2], 64, 64)
   for j=1,#sets[i] do
      finsets[#finsets+1] = sets[i][j]:float()
   end
end
finimage = image.toDisplayTensor{input=finsets, scaleeach=true, nrow=12}
image.save('/home/soumith/public_html/' .. tt .. '.png', finimage)
