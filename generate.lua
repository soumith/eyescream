require 'nn'
require 'image'

numImages = 128

m = {}
m[1] = torch.load('8x8.t7')
m[2] = torch.load('8x14.t7')
m[3] = torch.load('14x28.t7')

one_hot = torch.zeros(10) -- one-hot coding for class-conditional
classes = {'airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
one_hot[1] = 1 -- change this index to whichever class you want. For example horse = 8

set = {}
for i=1,numImages do
   noise = {} -- noise vectors
   noise[1] = torch.zeros(1,100):uniform(-1,1)
   noise[2] = torch.zeros(1,1,14,14):uniform(-1,1)
   noise[3] = torch.zeros(1,1,28,28):uniform(-1,1)

   o = {} -- outputs
   o[1] = m[1]:forward({noise[1], one_hot:view(1,10)})
   o[1] = image.scale(o[1], 14, 14):view(1,3,14,14)

   o[2] = m[2]:forward({noise[2], one_hot:view(1,10,1,1), o[1]})
   o[2] = o[2] + o[1]
   o[2] = image.scale(o[2][1], 28, 28):view(1,3,28,28)

   o[3] = m[3]:forward({noise[3], one_hot:view(1,10,1,1), o[2]})
   o[3] = o[3] + o[2]
   set[#set+1] = o[3][1]
end

img = image.toDisplayTensor{input=set, scaleeach=true, nrow=8}
if pcall(require, 'qttorch') then 
   image.display({image=img, zoom=2})
end
image.save('eyescream.png', img)
