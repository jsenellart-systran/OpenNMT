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
  self.criterion.sizeAverage = false

end

function SiameseRNN.load(args, models, _)
  local self = torch.factory('SiameseRNN')()

  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.utils.ExtendedCmdLine.getModuleOpts(args, options))

  self.models.encoder1 = onmt.Factory.loadEncoder(models.encoder1)
  self.models.encoder2 = onmt.Factory.loadEncoder(models.encoder2)
  self.models.comparator = onmt.ManhattanDistance(true)
  self.criterion = nn.MSECriterion()
  self.criterion.sizeAverage = false

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

function SiameseRNN:getOutputLabelsCount(batch)
  return batch.size
end

function SiameseRNN:forwardComputeLoss(batch)
  local _, context1 = self.models.encoder1:forward(batch)
  batch:switchInput()
  local _, context2 = self.models.encoder2:forward(batch)
  batch:switchInput()
  local diff = self.models.comparator:forward({context1[{{},-1,{}}], context2[{{},-1,{}}]})
  local ref = (batch:getTargetInput(2)-5):float()
  return self.criterion:forward(diff, ref)
end

function SiameseRNN:trainNetwork(batch)
  local _, context1 = self.models.encoder1:forward(batch)
  batch:switchInput()
  local _, context2 = self.models.encoder2:forward(batch)
  local diff = self.models.comparator:forward({context1[{{},-1,{}}], context2[{{},-1,{}}]})
  local ref = (batch:getTargetInput(2)-5):float()
  local loss = self.criterion:forward(diff, ref)
  local decComparatorOut = self.criterion:backward(diff, ref)
  decComparatorOut:div(batch.totalSize)
  local decEncoderOut = self.models.comparator:backward({context1[{{},-1,{}}], context2[{{},-1,{}}]}, decComparatorOut)
  context2:zero()[{{},-1,{}}]:copy(decEncoderOut[2])
  self.models.encoder2:backward(batch, nil, context2)
  batch:switchInput()
  context1:zero()[{{},-1,{}}]:copy(decEncoderOut[1])
  self.models.encoder1:backward(batch, nil, context1)
  return loss
end

return SiameseRNN
