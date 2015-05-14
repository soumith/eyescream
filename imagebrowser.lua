disp=require 'display'
require 'image'
require 'paths'


local function customsort(a,b)
   if tonumber(a:sub(#a-a:reverse():find('_')+2, #a-4))
   > tonumber(b:sub(#b-b:reverse():find('_')+2, #b-4)) then
      return false
   else
      return true
   end
end

imbrowse = function(folder)
   assert(paths.dirp(folder))
   local list = {}
   for p in paths.files(folder) do
      if p:find('.png') then
         local path = paths.concat(folder, p)
         list[#list+1] = path
      end
   end
   table.sort(list, customsort)
   for k,path in pairs(list) do
      print(path)
      local im = image.load(path)
      disp.image(im, {win=10, width=800})
      print('Press enter for next image')
      local s = io.read("*l")
   end
end

assert(arg[1], 'give folderpath as first argument')
imbrowse(arg[1])
