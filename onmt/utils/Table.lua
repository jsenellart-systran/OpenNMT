local tds = require('tds')

--[[ Append table `src` to `dst`. ]]
local function append(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

--[[ Reorder table `tab` based on the `index` array. ]]
local function reorder(tab, index, cdata)
  local newTab
  if cdata then
    newTab = tds.Vec()
    newTab:resize(#tab)
  else
    newTab = {}
  end

  for i = 1, #tab do
    newTab[i] = tab[index[i]]
  end

  return newTab
end

--[[ Recursively div table of tensor by constant ]]
local function div(tab, c)
  for j = 1, #tab do
    if type(tab[j]) == 'table' then
      div(tab[j], c)
    else
      tab[j]:div(c)
    end
  end
end

return {
  reorder = reorder,
  append = append,
  div = div
}
