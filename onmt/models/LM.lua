--[[ Language Model. ]]
require 'onmt.models.Model'
local LM, parent = torch.class('onmt.Model.LM', 'onmt.Model')

local LM_options = {
  {'-layers', 2, [[Number of layers in the RNN encoder/decoder]]},
  {'-rnn_size', 500, [[Size of RNN hidden states]]},
  {'-rnn_type', 'LSTM', [[Type of RNN cell: LSTM, GRU]]},
  {'-word_vec_size', '500', [[Comma-separated list of embedding sizes: word[,feat1,feat2,...].]]},
  {'-feat_merge', 'concat', [[Merge action for the features embeddings.]],
                     {enum={'concat','sum'}}},
  {'-feat_vec_exponent', 0.7, [[When using concatenation, if the feature takes N values
                                        then the embedding dimension will be set to N^exponent]]},
  {'-feat_vec_size', 20, [[When using sum, the common embedding size of the features]]},
  {'-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]]},
  {'-residual', false, [[Add residual connections between RNN layers.]]},
  {'-brnn', false, [[Use a bidirectional encoder]]},
  {'-brnn_merge', 'sum', [[Merge action for the bidirectional hidden states.]],
                     {enum={'concat','sum'}}},
  {'-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]],
                         {valid=onmt.ExtendedCmdLine.fileNullOrExists}},
  {'-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]]},
  {'-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]]}
}

function LM.declareOpts(cmd)
  cmd:setCmdLineOptions(LM_options, "Language Model")
  onmt.AdaptiveSoftMax.declareOpts(cmd)
end

function LM:__init(args, dicts)
  parent.__init(self, args)
  onmt.utils.Table.merge(self.args, onmt.ExtendedCmdLine.getModuleOpts(args, LM_options))
  self.args.adaptive_softmax = args.adaptive_softmax

  -- encoder word_vec_size is in src_word_vec_size
  self.args.src_word_vec_size = args.word_vec_size
  self.args.word_vec_size = 0

  self.models.encoder = onmt.Factory.buildWordEncoder(self.args, dicts.src)

  local adaptive_softmax_cutoff
  if self.args.adaptive_softmax and self.args.adaptive_softmax ~= '' then
    self.adaptive_softmax_cutoff = loadstring(" return "..self.args.adaptive_softmax)()
    table.insert(self.adaptive_softmax_cutoff, dicts.src.words:size())
    _G.logger:info(" * using adaptive_softmax_cutoff: {"..table.concat(self.adaptive_softmax_cutoff,',').."}")
  end

  if #dicts.src.features > 0 then
    self.models.generator = onmt.FeaturesGenerator.new(self.args.rnn_size,
                                                       dicts.src.words:size(),
                                                       dicts.src.features)
  else
    self.models.generator = onmt.Generator.new(self.args.rnn_size, dicts.src.words:size(), self.adaptive_softmax_cutoff)
  end

  self.EOS_vector_model = torch.LongTensor(args.max_batch_size):fill(onmt.Constants.EOS)
end

function LM.load()
  error("loading a language model is not yet supported")
end

-- Returns model name.
function LM.modelName()
  return "Language"
end

-- Returns expected dataMode.
function LM.dataType()
  return "monotext"
end

-- batch fields for language model
function LM.batchInit()
  return {
           size = 1,
           sourceLength = 0
         }
end

function LM.batchAggregate(batchA, batch)
  batchA.sourceLength = batchA.sourceLength + batch.sourceLength * batch.size
  return batchA
end

function LM:forwardComputeLoss(batch, criterion)
  local _, context = self.models.encoder:forward(batch)
  local EOS_vector = self.EOS_vector_model:narrow(1, 1, batch.size)
  onmt.utils.Cuda.convert(EOS_vector)
  local loss = 0
  for t = 1, batch.sourceLength do
    -- LM is supposed to predict the following word.
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    else
      output = EOS_vector
    end
    if self.adaptive_softmax_cutoff then
      self.models.generator.adaptive_softmax:setTarget(output)
    end

    local genOutputs = self.models.generator:forward(context:select(2, t))

    -- Same format with and without features.
    if torch.type(output) ~= 'table' then output = { output } end
    loss = loss + criterion:forward(genOutputs, output)
  end
  return loss
end

function LM:buildCriterion(dataset)
  return onmt.Criterion.new(dataset.dicts.src.words:size(),
                            dataset.dicts.src.features,
                            self.adaptive_softmax_cutoff)
end

function LM:countTokens(batch)
  return batch.sourceLength*batch.size
end

function LM:trainNetwork(batch, criterion, doProfile)
  local loss = 0

  if doProfile then _G.profiler:start("encoder.fwd") end
  local _, context = self.models.encoder:forward(batch)
  if doProfile then _G.profiler:stop("encoder.fwd") end

  local gradContexts = torch.Tensor(batch.size, batch.sourceLength, self.args.rnn_size)
  gradContexts = onmt.utils.Cuda.convert(gradContexts)
  -- for each word of the sentence, generate target
  for t = 1, batch.sourceLength do
    -- LM is supposed to predict following word
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    else
      output = self.EOS_vector_model:narrow(1, 1, batch.size)
    end
    if self.adaptive_softmax_cutoff then
      self.models.generator.adaptive_softmax:setTarget(output)
    end

    if doProfile then _G.profiler:start("generator.fwd") end
    local genOutputs = self.models.generator:forward(context:select(2,t))
    if doProfile then _G.profiler:stop("generator.fwd") end

    -- same format with and without features
    if torch.type(output) ~= 'table' then output = { output } end

    if doProfile then _G.profiler:start("criterion.fwd") end
    loss = loss + criterion:forward(genOutputs, output)
    if doProfile then _G.profiler:stop("criterion.fwd") end

    -- backward
    if doProfile then _G.profiler:start("criterion.bwd") end
    local genGradOutput = criterion:backward(genOutputs, output)
    if doProfile then _G.profiler:stop("criterion.bwd") end

    onmt.utils.Table.div(genGradOutput, batch.totalSize)

    if doProfile then _G.profiler:start("generator.bwd") end
    gradContexts[{{}, t}]:copy(self.models.generator:backward(context:select(2, t), genGradOutput))
    if doProfile then _G.profiler:stop("generator.bwd") end

  end

  if doProfile then _G.profiler:start("encoder.bwd") end
  self.models.encoder:backward(batch, nil, gradContexts)
  if doProfile then _G.profiler:stop("encoder.bwd") end

  return loss
end
