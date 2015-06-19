require 'nn'

m1 = torch.load('8x8.t7')
m2 = torch.load('8x14.t7')
m3 = torch.load('14x28.t7')

print(m1)
print(m2)
print(m3)
