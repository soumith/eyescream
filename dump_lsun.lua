opt = {}
opt.dataset = 'lsun'
opt.nDonkeys = 0
opt.save = 'dumplsuns'
opt.loadSize = 256
opt.fineSize = 256

paths.dofile('data.lua')
os.execute('mkdir -p ' .. opt.save)

local class = 0
local set = {}
for i=1,testLoader:size(),300 do
   class = class + 1
   set[class] = set[class] or {}
   for j=1,300 do
      local filename = 'imagelarge' .. class .. '_num_' .. j .. '.png'
      print(filename)
      local img,lab = testLoader:get(i+j-1, i+j-1)
      print(#img)
      print(i, j, class, lab[1])
      assert(lab[1] == class)
      image.save(opt.save .. '/' .. filename, img[1])
      table.insert(set[class], img[1])
   end
end

for i=1,10 do
   local img = image.toDisplayTensor{input=set[i], scaleeach=true, nrow=8}
   image.save('sheet_' .. i .. '.png', img)
end
