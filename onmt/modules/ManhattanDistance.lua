--[[ nn module to compute Manhattan Distance between 2 tensor
--]]
local ManhattanDistance, parent = torch.class('onmt.ManhattanDistance', 'onmt.Network')

--[[ Construct an ManhattanDistance layer.
]]
function ManhattanDistance:__init(args)
  parent.__init(self, self:_buildModel())
end

--[[
--]]
function ManhattanDistance:_buildModel()
  local inputs = {}
  local states = {}

  local x = nn.Identity()()
  table.insert(inputs, x)
  local y = nn.Identity()()
  table.insert(inputs, y)

  local outputs = nn.Sum(1, 1)(nn.Abs()(nn.CSubTable()({x, y})))

  return nn.gModule(inputs, { outputs })
end

