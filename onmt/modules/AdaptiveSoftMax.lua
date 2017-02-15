--[[ If available Integrate Adaptive Module
]]

local ret1, _ = pcall(require, 'onmt.modules.adaptive-softmax.AdaptiveSoftMax')
local ret2, _ = pcall(require, 'onmt.modules.adaptive-softmax.AdaptiveLoss')

if ret1 and ret2 then
  -- modules are available - inherit from them
  local AdaptiveSoftMax = torch.class('onmt.AdaptiveSoftMax', 'nn.AdaptiveSoftMax')
  torch.class('onmt.AdaptiveLoss', 'nn.AdaptiveLoss')

  function AdaptiveSoftMax.declareOpts(cmd)
    cmd:option('-adaptive_softmax', '', [[Use Adaptive SoftMax - need to be given a vocabulary frequence partition like {2000,1000}.
      If value is {} - works a NCE.]])
    cmd:option('-adaptive_softmax_capacity_reduction', 4, [[Capacity reduction of cluster capacity.]],
                    {valid=onmt.ExtendedCmdLine.floatRange(1)})
  end

else
  -- declare empty class
  local AdaptiveSoftMax = torch.class('onmt.AdaptiveSoftMax')

  function AdaptiveSoftMax.declareOpts(cmd)
  end

end
