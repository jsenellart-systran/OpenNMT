--[[ Language Model. ]]
local SiameseRNN, parent = torch.class('SiameseRNN', 'Model')

local options = {
  {
    '-word_vec_size', { 500 },
    [[List of embedding sizes: `word[ feat1[ feat2[ ...] ] ]`.]],
    {
      structural = 0
    }
  },
  {
    '-pre_word_vecs_enc', '',
    [[Path to pretrained word embeddings on the encoder side serialized as a Torch tensor.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileNullOrExists,
      init_only = true
    }
  },
  {
    '-fix_word_vecs_enc', false,
    [[Fix word embeddings on the encoder side.]],
    {
      structural = 1
    }
  }
}

function SiameseRNN.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Siamese RNN')
  onmt.Encoder.declareOpts(cmd)
  onmt.Factory.declareOpts(cmd)
end

function SiameseRNN:__init(args, dicts)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder1 = onmt.Factory.buildWordEncoder(args, dicts.src)
  self.models.encoder2 = self.models.encoder1:clone('weight', 'bias', 'gradWeight', 'gradBias')

  self.models.comparator = onmt.ManhattanDistance(true)

  self.criterion = nn.MSECriterion()
end

function SiameseRNN.load(args, models, _)
  local self = torch.factory('SiameseRNN')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder = onmt.Factory.loadEncoder(models.encoder)
  self.models.comparator = onmt.ManhattanDistance(true)
  self.criterion = nn.MSECriterion()

  return self
end

-- Returns model name.
function SiameseRNN.modelName()
  return 'SiameseRNN'
end

-- Returns expected dataMode.
function SiameseRNN.dataType()
  return 'tritext'
end

function SiameseRNN:enableProfiling()
  _G.profiler.addHook(self.models.encoder1, 'encoder1')
  _G.profiler.addHook(self.models.encoder2, 'encoder2')
  _G.profiler.addHook(self.models.comparator, 'comparator')
  _G.profiler.addHook(self.criterion, 'criterion')
end

local function switchInput(batch)
  batch.sourceInput, batch.sourceInput2 = batch.sourceInput2, batch.sourceInput
  batch.sourceInputRev, batch.sourceInputRev2 = batch.sourceInputRev2, batch.sourceInputRev
  batch.sourceSize, batch.sourceSize2 = batch.sourceSize2, batch.sourceSize
  batch.sourceLength, batch.sourceLength2 = batch.sourceLength2, batch.sourceLength
  batch.uneven, batch.uneven2 = batch.uneven2, batch.uneven
  batch.inputVectors, batch.inputVectors2 = batch.inputVectors2, batch.inputVectors
  batch.sourceInputFeatures, batch.sourceInputFeatures2 = batch.sourceInputFeatures2, batch.sourceInputFeatures
  batch.sourceInputRevFeatures, batch.sourceInputRevFeatures2 = batch.sourceInputRevFeatures2, batch.sourceInputRevFeatures
  batch.padTensor, batch.padTensor2= batch.padTensor2, batch.padTensor
  batch.sourceInputPadLeft, batch.sourceInputPadLeft2 = batch.sourceInputPadLeft2, batch.sourceInputPadLeft
  batch.sourceInputRevPadLeft, batch.sourceInputRevPadLeft2 = batch.sourceInputRevPadLeft2, batch.sourceInputRevPadLeft
end

function SiameseRNN:getOutputLabelsCount(batch)
  return batch.size
end

function SiameseRNN:forwardComputeLoss(batch)
  local _, context1 = self.models.encoder1:forward(batch)
  switchInput(batch)
  local _, context2 = self.models.encoder2:forward(batch)
  switchInput(batch)
  local diff = self.models.comparator:forward({context1[{{},-1,{}}], context2[{{},-1,{}}]})
  local ref = (batch:getTargetInput(2)-5):float()
  return self.criterion:forward(diff, ref)
end

function SiameseRNN:trainNetwork(batch)
  local _, context1 = self.models.encoder1:forward(batch)
  switchInput(batch)
  local _, context2 = self.models.encoder2:forward(batch)
  local diff = self.models.comparator:forward({context1[{{},-1,{}}], context2[{{},-1,{}}]})
  local ref = (batch:getTargetInput(2)-5):float()
  local loss = self.criterion:forward(diff, ref)
  local decComparatorOut = self.criterion:backward(diff, ref)
  decComparatorOut:div(batch.totalSize)
  local decEncoderOut = self.models.comparator:backward({context1[{{},-1,{}}], context2[{{},-1,{}}]}, decComparatorOut)
  context2:zero()[{{},-1,{}}]:copy(decEncoderOut[2])
  self.models.encoder2:backward(batch, nil, context2)
  switchInput(batch)
  context1:zero()[{{},-1,{}}]:copy(decEncoderOut[1])
  self.models.encoder1:backward(batch, nil, context1)
  return loss
end

return SiameseRNN
