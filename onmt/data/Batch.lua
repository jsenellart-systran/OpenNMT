--[[ Return the maxLength, sizes, and non-zero count
  of a batch of `seq`s ignoring `ignore` words.
--]]
local function getLength(seq, ignore)
  local sizes = torch.IntTensor(#seq):zero()
  local max = 0
  local uneven = false

  for i = 1, #seq do
    local len = seq[i]:size(1)
    if ignore ~= nil then
      len = len - ignore
    end
    if max == 0 or len > max then
      max = len
    end
    if i > 1 and sizes[i - 1] ~= len then
      uneven = true
    end
    sizes[i] = len
  end
  return max, sizes, uneven
end

--[[ Data management and batch creation.

Batch interface reference [size]:

  * size: number of sentences in the batch [1]
  * sourceLength: max length in source batch [1]
  * sourceSize:  lengths of each source [batch x 1]
  * sourceInput:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * sourceInputFeatures: table of source features sequences
  * sourceInputRev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * sourceInputRevFeatures: table of reversed source features sequences
  * targetLength: max length in source batch [1]
  * targetSize: lengths of each source [batch x 1]
  * targetInput: input idx's of target (SABCDEPPPPPP) [batch x max]
  * targetInputFeatures: table of target input features sequences
  * targetOutput: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * targetOutputFeatures: table of target output features sequences

 TODO: change name of size => maxlen
--]]


--[[ A batch of sentences to translate and targets. Manages padding,
  features, and batch alignment (for efficiency).

  Used by the decoder and encoder objects.
--]]
local Batch = torch.class('Batch')

--[[ Create a batch object.

Parameters:

  * `src` - 2D table of source batch indices or prebuilt source batch vectors
  * `srcFeatures` - 2D table of source batch features (opt)
  * `tgt` - 2D table of target batch indices
  * `tgtFeatures` - 2D table of target batch features (opt)
  * `src2` - 2D table of second source batch indices (opt)
  * `src2Features` - 2D table of second source batch features (opt)
--]]
function Batch:__init(src, srcFeatures, tgt, tgtFeatures, src2, src2Features)
  src = src or {}
  srcFeatures = srcFeatures or {}
  tgtFeatures = tgtFeatures or {}
  src2Features = src2Features or {}

  if tgt ~= nil then
    assert(#src == #tgt, "source and target must have the same batch size")
  end
  if src2 ~= nil then
    assert(#src == #src2, "source and second source must have the same batch size")
  end

  self.size = #src
  self.totalSize = self.size -- updated when this batch is part of a larger one (data parallelism).

  self.sourceLength, self.sourceSize, self.uneven = getLength(src)

  -- if input vectors (speech for instance)
  self.inputVectors = #src > 0 and src[1]:dim() > 1

  local sourceSeq = torch.LongTensor(self.sourceLength, self.size):fill(onmt.Constants.PAD)

  if not self.inputVectors then
    self.sourceInput = sourceSeq:clone()
    self.sourceInputRev = sourceSeq:clone()
    -- will be used to return extra padded value
    self.padTensor = torch.LongTensor(self.size):fill(onmt.Constants.PAD)
  else
    self.sourceInput = torch.Tensor(self.sourceLength, self.size, src[1]:size(2))
    self.sourceInputRev = torch.Tensor(self.sourceLength, self.size, src[1]:size(2))
    self.padTensor = torch.Tensor(self.size, src[1]:size(2)):zero()
  end

  self.sourceInputFeatures = {}
  self.sourceInputRevFeatures = {}

  if #srcFeatures > 0 then
    for _ = 1, #srcFeatures[1] do
      table.insert(self.sourceInputFeatures, sourceSeq:clone())
      table.insert(self.sourceInputRevFeatures, sourceSeq:clone())
    end
  end

  if src2 ~= nil then
    self.sourceLength2, self.sourceSize2, self.uneven2 = getLength(src2)

    -- if input vectors (speech for instance)
    self.inputVectors2 = #src2 > 0 and src2[1]:dim() > 1

    local sourceSeq2 = torch.LongTensor(self.sourceLength2, self.size):fill(onmt.Constants.PAD)

    if not self.inputVectors2 then
      self.sourceInput2 = sourceSeq2:clone()
      self.sourceInputRev2 = sourceSeq2:clone()
      -- will be used to return extra padded value
      self.padTensor2 = torch.LongTensor(self.size):fill(onmt.Constants.PAD)
    else
      self.sourceInput2 = torch.Tensor(self.sourceLength2, self.size, src2[1]:size(2))
      self.sourceInputRev2 = torch.Tensor(self.sourceLength2, self.size, src2[1]:size(2))
      self.padTensor2 = torch.Tensor(self.size, src2[1]:size(2)):zero()
    end

    self.sourceInputFeatures2 = {}
    self.sourceInputRevFeatures2 = {}

    if #src2Features > 0 then
      for _ = 1, #src2Features[1] do
        table.insert(self.sourceInputFeatures2, sourceSeq:clone())
        table.insert(self.sourceInputRevFeatures2, sourceSeq:clone())
      end
    end
  end

  if tgt ~= nil then
    self.targetLength, self.targetSize = getLength(tgt, 1)

    local targetSeq = torch.LongTensor(self.targetLength, self.size):fill(onmt.Constants.PAD)
    self.targetInput = targetSeq:clone()
    self.targetOutput = targetSeq:clone()

    self.targetInputFeatures = {}
    self.targetOutputFeatures = {}

    if #tgtFeatures > 0 then
      for _ = 1, #tgtFeatures[1] do
        table.insert(self.targetInputFeatures, targetSeq:clone())
        table.insert(self.targetOutputFeatures, targetSeq:clone())
      end
    end
  end

  for b = 1, self.size do
    local sourceOffset = self.sourceLength - self.sourceSize[b] + 1
    local sourceInput = src[b]
    local sourceInputRev = src[b]:index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())

    -- Source input is left padded [PPPPPPABCDE] .
    self.sourceInput[{{sourceOffset, self.sourceLength}, b}]:copy(sourceInput)
    self.sourceInputPadLeft = true

    -- Rev source input is right padded [EDCBAPPPPPP] .
    self.sourceInputRev[{{1, self.sourceSize[b]}, b}]:copy(sourceInputRev)
    self.sourceInputRevPadLeft = false

    for i = 1, #self.sourceInputFeatures do
      local sourceInputFeatures = srcFeatures[b][i]
      local sourceInputRevFeatures = srcFeatures[b][i]:index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())

      self.sourceInputFeatures[i][{{sourceOffset, self.sourceLength}, b}]:copy(sourceInputFeatures)
      self.sourceInputRevFeatures[i][{{1, self.sourceSize[b]}, b}]:copy(sourceInputRevFeatures)
    end

    if src2 ~= nil then
      local sourceOffset2 = self.sourceLength2 - self.sourceSize2[b] + 1
      local sourceInput2 = src2[b]
      local sourceInputRev2 = src2[b]:index(1, torch.linspace(self.sourceSize2[b], 1, self.sourceSize2[b]):long())

      -- Source input is left padded [PPPPPPABCDE] .
      self.sourceInput2[{{sourceOffset, self.sourceLength2}, b}]:copy(sourceInput2)
      self.sourceInputPadLeft2 = true

      -- Rev source input is right padded [EDCBAPPPPPP] .
      self.sourceInputRev2[{{1, self.sourceSize2[b]}, b}]:copy(sourceInputRev2)
      self.sourceInputRevPadLeft2 = false

      for i = 1, #self.sourceInputFeatures2 do
        local sourceInputFeatures2 = src2Features[b][i]
        local sourceInputRevFeatures2 = src2Features[b][i]:index(1, torch.linspace(self.sourceSize2[b], 1, self.sourceSize2[b]):long())

        self.sourceInputFeatures2[i][{{sourceOffset2, self.sourceLength2}, b}]:copy(sourceInputFeatures2)
        self.sourceInputRevFeatures2[i][{{1, self.sourceSize2[b]}, b}]:copy(sourceInputRevFeatures2)
      end
    end

    if tgt ~= nil then
      -- Input: [<s>ABCDE]
      -- Ouput: [ABCDE</s>]
      local targetLength = tgt[b]:size(1) - 1

      local targetInput = tgt[b]:narrow(1, 1, targetLength)
      local targetOutput = tgt[b]:narrow(1, 2, targetLength)

      -- Target is right padded [<S>ABCDEPPPPPP] .
      self.targetInput[{{1, targetLength}, b}]:copy(targetInput)
      self.targetOutput[{{1, targetLength}, b}]:copy(targetOutput)

      for i = 1, #self.targetInputFeatures do
        local targetInputFeatures = tgtFeatures[b][i]:narrow(1, 1, targetLength)
        local targetOutputFeatures = tgtFeatures[b][i]:narrow(1, 2, targetLength)

        self.targetInputFeatures[i][{{1, targetLength}, b}]:copy(targetInputFeatures)
        self.targetOutputFeatures[i][{{1, targetLength}, b}]:copy(targetOutputFeatures)
      end
    end
  end
end

--[[ Set source input directly,

Parameters:

  * `sourceInput` - a Tensor of size (sequence_length, batch_size, feature_dim)
  ,or a sequence of size (sequence_length, batch_size). Be aware that sourceInput is not cloned here.

--]]
function Batch:setSourceInput(sourceInput)
  assert (sourceInput:dim() >= 2, 'The sourceInput tensor should be of size (seq_len, batch_size, ...)')
  self.size = sourceInput:size(2)
  self.sourceLength = sourceInput:size(1)
  self.sourceInputFeatures = {}
  self.sourceInputRevReatures = {}
  self.sourceInput = sourceInput
  self.sourceInputRev = self.sourceInput:index(1, torch.linspace(self.sourceLength, 1, self.sourceLength):long())
  return self
end

--[[ Set target input directly.

Parameters:

  * `targetInput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Constants.PAD. Be aware that targetInput is not cloned here.
--]]
function Batch:setTargetInput(targetInput)
  assert (targetInput:dim() == 2, 'The targetInput tensor should be of size (seq_len, batch_size)')
  self.targetInput = targetInput
  self.size = targetInput:size(2)
  self.totalSize = self.size
  self.targetLength = targetInput:size(1)
  self.targetInputFeatures = {}
  self.targetSize = torch.sum(targetInput:transpose(1,2):ne(onmt.Constants.PAD), 2):view(-1):double()
  return self
end

--[[ Set target output directly.

Parameters:

  * `targetOutput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Constants.PAD.  Be aware that targetOutput is not cloned here.
--]]
function Batch:setTargetOutput(targetOutput)
  assert (targetOutput:dim() == 2, 'The targetOutput tensor should be of size (seq_len, batch_size)')
  self.targetOutput = targetOutput
  self.targetOutputFeatures = {}
  return self
end

local function addInputFeatures(inputs, featuresSeq, t)
  local features = {}
  for j = 1, #featuresSeq do
    local feat
    if t > featuresSeq[j]:size(1) then
      feat = onmt.Constants.PAD
    else
      feat = featuresSeq[j][t]
    end
    table.insert(features, feat)
  end
  if #features > 1 then
    table.insert(inputs, features)
  else
    onmt.utils.Table.append(inputs, features)
  end
end

--[[ Get source input batch at timestep `t`. --]]
function Batch:getSourceInput(t)
  local inputs

  -- If a regular input, return word id, otherwise a table with features.
  if t > self.sourceInput:size(1) then
    inputs = self.padTensor
  else
    inputs = self.sourceInput[t]
  end

  if #self.sourceInputFeatures > 0 then
    inputs = { inputs }
    addInputFeatures(inputs, self.sourceInputFeatures, t)
  end

  return inputs
end

--[[ Get source input batch at timestep `t`. --]]
function Batch:getSourceInput2(t)
  local inputs

  -- If a regular input, return word id, otherwise a table with features.
  if t > self.sourceInput2:size(1) then
    inputs = self.padTensor2
  else
    inputs = self.sourceInput2[t]
  end

  if #self.sourceInputFeatures2 > 0 then
    inputs = { inputs }
    addInputFeatures(inputs, self.sourceInputFeatures2, t)
  end

  return inputs
end

--[[ Get target input batch at timestep `t`. --]]
function Batch:getTargetInput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local inputs = self.targetInput[t]

  if #self.targetInputFeatures > 0 then
    inputs = { inputs }
    addInputFeatures(inputs, self.targetInputFeatures, t)
  end

  return inputs
end

--[[ Get target output batch at timestep `t` (values t+1). --]]
function Batch:getTargetOutput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local outputs = { self.targetOutput[t] }

  for j = 1, #self.targetOutputFeatures do
    table.insert(outputs, self.targetOutputFeatures[j][t])
  end

  return outputs
end

return Batch
