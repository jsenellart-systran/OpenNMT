require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('compare.lua')

local options = {
  {
    '-model', '',
    [[Comparison Model.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-src1', '',
    [[Source sequences to compare.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-src2', '',
    [[Source sequences to compare.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  },
  {
    '-tgt', '',
    [[Optional true target sequences.]]
  },
  {
    '-output', 'pred.txt',
    [[Output file.]]
  }
}

cmd:setCmdLineOptions(options, 'Data')

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

cmd:text('')
cmd:text('Other options')
cmd:text('')

cmd:option('-time', false, [[Measure average comparison time.]])

local function buildInput(tokens)
  local data = {}
  local words, features = onmt.utils.Features.extract(tokens)

  data.words = words

  if #features > 0 then
    data.features = features
  end
  return data
end

local function buildData(src1, src2, dicts)
  local srcData1 = {}
  srcData1.words = {}
  srcData1.features = {}

  local srcData2 = {}
  srcData2.words = {}
  srcData2.features = {}

  local ignored = {}
  local indexMap = {}
  local index = 1

  for b = 1, #src1 do
    if (src1[b].words and #src1[b].words == 0) or (src2[b].words and #src2[b].words == 0) then
      table.insert(ignored, b)
    else
      indexMap[index] = b
      index = index + 1

      if dicts.src then
        table.insert(srcData1.words,
                   dicts.src.words:convertToIdx(src1[b].words, onmt.Constants.UNK_WORD))
        table.insert(srcData2.words,
                   dicts.src.words:convertToIdx(src2[b].words, onmt.Constants.UNK_WORD))
        if #dicts.src.features > 0 then
          table.insert(srcData1.features,
                       onmt.utils.Features.generateSource(dicts.src.features, src1[b].features))
          table.insert(srcData2.features,
                       onmt.utils.Features.generateSource(dicts.src.features, src2[b].features))
        end
      else
        table.insert(srcData1.vectors, onmt.utils.Cuda.convert(src1[b].vectors))
        table.insert(srcData2.vectors, onmt.utils.Cuda.convert(src2[b].vectors))
      end
    end
  end

  return onmt.data.Dataset.new(srcData1, nil, srcData2), ignored, indexMap
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  _G.profiler = onmt.utils.Profiler.new()

  onmt.utils.Cuda.init(opt)

  local srcReader1 = onmt.utils.FileReader.new(opt.src1)
  local srcReader2 = onmt.utils.FileReader.new(opt.src2)

  local checkpoint = torch.load(opt.model)

  assert(checkpoint.options.model_type == 'siamese')

  local model = onmt.SiameseRNN.load(opt, checkpoint.models, checkpoint.dicts)

  onmt.utils.Cuda.convert(model)

  model:evaluate()

  local dicts = checkpoint.dicts

  local outFile = io.open(opt.output, 'w')

  local withGoldScore = opt.tgt:len() > 0

  if withGoldScore then
    goldReader = onmt.utils.FileReader.new(opt.tgt, opt.idx_files)
    goldBatch = {}
  end

  local timer
  if opt.time then
    timer = torch.Timer()
    timer:stop()
    timer:reset()
  end

  local sentId = 1
  local score = 0

  while true do
    local srcSeq1 = srcReader1:next()
    local srcSeq2 = srcReader2:next()
    if not srcSeq1 or not srcSeq2 then
      break
    end

    local goldOutput
    if withGoldScore then
      local goldOutputSeq
      goldOutputSeq = goldReader:next()
      assert(#goldOutputSeq==1 and (goldOutputSeq[1]=="0" or goldOutputSeq[1]=="1"))
      goldOutput = tonumber(goldOutputSeq[1])
    end

    local srcBatch1 = {}
    local srcBatch2 = {}
    table.insert(srcBatch1, buildInput(srcSeq1))
    table.insert(srcBatch2, buildInput(srcSeq2))

    local data = buildData(srcBatch1, srcBatch2, dicts)

     if data:batchCount() > 0 then
      local batch = onmt.utils.Cuda.convert(data:getBatch())

      local _, context1 = model.models.encoder1:forward(batch)
      batch:switchInput()

      local _, context2 = model.modelClones.encoder2:forward(batch)
      local diff = model.models.comparator:forward({context1[{{},-1,{}}], context2[{{},-1,{}}]})

      local probability = diff[1]

      outFile:write(probability..'\n')

      _G.logger:info('SENT1 %d: %s', sentId, table.concat(srcSeq1, " "))
      _G.logger:info('SENT2 %d: %s', sentId, table.concat(srcSeq2, " "))
      _G.logger:info('PRED %d: %f', sentId, probability)

      if goldOutput then
        _G.logger:info('GOLD %d: %d', sentId, goldOutput)
        if probability == 1 then probability = 0.999 end
        local loss = - goldOutput * math.log(probability) - (1-goldOutput) * math.log(1-probability)
        _G.logger:info('GOLD SCORE %d: %f', sentId, loss)
        score = score + loss
      end

      sentId = sentId + 1
    end
  end

  if withGoldScore then
    _G.logger:info("Log Loss = %f\n", score/sentId)
  end

  if opt.time then
    local time = timer:time()
    local sentenceCount = sentId-1
    _G.logger:info("Average sentence translation time (in seconds):\n")
    _G.logger:info("avg real\t" .. time.real / sentenceCount .. "\n")
    _G.logger:info("avg user\t" .. time.user / sentenceCount .. "\n")
    _G.logger:info("avg sys\t" .. time.sys / sentenceCount .. "\n")
  end

  outFile:close()
  _G.logger:shutDown()
end

main()
