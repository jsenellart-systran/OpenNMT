require('nngraph')

--[[ Fixed attention

--]]
local FixedAttention, parent = torch.class('onmt.FixedAttention', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function FixedAttention:__init(dim)
  self.fixedAttnTensor = torch.Tensor()
  -- simulate a gModule returning the fixedAttnTensor
  local inputs = nn.Identity()()
  self.attnTensorModule = nn.gModule({inputs},{inputs})
  self.attnTensorModule.gradInput = torch.Tensor()
  self.attnTensorModule.updateOutput = function(input)
    return self.attnTensorModule:runForwardFunction(function()
      print('==',self.fixedAttnTensor)
      return self.fixedAttnTensor end,
    input)
  end
  self.attnTensorModule.updateGradInput = function(m, input, gradOutput)
    m.outnode.data.gradOutput = m.outnode.data.gradOutput or {}
    m.outnode.data.gradOutput[1] = gradOutput
    return m.gradInput:resize(input:size()):zero()
  end
  self.attnTensorModule.accGradParameters = function() end
  parent.__init(self, self:_buildModel(dim))
end

function FixedAttention:_buildModel(dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local targetT = nn.Linear(dim, dim, false)(inputs[1]) -- batchL x dim
  local context = inputs[2] -- batchL x sourceTimesteps x dim

  -- Set attention.
  local attn = nn.Replicate(1,2)(self.attnTensorModule(targetT)) -- batchL x 1 x sourceL

  -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput})
end
