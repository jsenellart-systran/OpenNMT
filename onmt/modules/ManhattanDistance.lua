--[[ nn module to compute Manhattan Distance between 2 tensor
--]]
local ManhattanDistance, parent = torch.class('onmt.ManhattanDistance', 'onmt.Network')

--[[ Construct an ManhattanDistance layer.
]]
function ManhattanDistance:__init(expNorm)
  parent.__init(self, self:_buildModel(expNorm))
end

--[[
--]]
function ManhattanDistance:_buildModel(expNorm)
  local inputs = {}

  local x = nn.Identity()()
  table.insert(inputs, x)
  local y = nn.Identity()()
  table.insert(inputs, y)

  local outputs = nn.Sum(1, 1)(nn.Abs()(nn.CSubTable()({x, y})))

  if expNorm then
    outputs = nn.Exp()(nn.MulConstant(-1)(outputs))
  end

  return nn.gModule(inputs, { outputs })
end

