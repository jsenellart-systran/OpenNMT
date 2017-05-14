require('onmt.init')

local tester = ...

local preprocessorTest = torch.TestSuite()

local dataDir = 'data'

local noFilter = function(_) return true end

local function buildPreprocessor(mode)
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.data.Preprocessor.declareOpts(cmd, mode)

  local commandLine
  if not mode or mode == 'bitext' then
    commandLine = {
      '-train_src', dataDir .. '/src-val-case.txt',
      '-train_tgt', dataDir .. '/tgt-val-case.txt',
      '-valid_src', dataDir .. '/src-test-case.txt',
      '-valid_tgt', dataDir .. '/tgt-test-case.txt'
    }
  elseif mode == 'monotext' then
    commandLine = {
      '-train', dataDir .. '/src-val-case.txt',
      '-valid', dataDir .. '/src-test-case.txt'
    }
  elseif mode == 'feattext' then
    commandLine = {
      '-train_src', dataDir .. '/sigtraintrig.srcfeat',
      '-train_tgt', dataDir .. '/sigtraintrig.tgt',
      '-valid_src', dataDir .. '/sigvaltrig.srcfeat',
      '-valid_tgt', dataDir .. '/sigvaltrig.tgt',
      '-idx_files',
      '-src_seq_length', 100
    }
  elseif mode == 'tritext' then
    commandLine = {
      '-train_src1', dataDir .. '/src-val-case.txt',
      '-train_src2', dataDir .. '/src-val.txt',
      '-train_tgt', dataDir .. '/tgt-val-case.txt',
      '-valid_src1', dataDir .. '/src-test-case.txt',
      '-valid_src2', dataDir .. '/src-test.txt',
      '-valid_tgt', dataDir .. '/tgt-test-case.txt'
      }
  end

  local opt = cmd:parse(commandLine)

  return onmt.data.Preprocessor.new(opt, mode), opt
end

local function makeDicts(srctgt, file)
  return onmt.data.Vocabulary.init(srctgt, file, '',  { 0 }, { 0 }, '', noFilter)
end

function preprocessorTest.bitext()
  local preprocessor, opt = buildPreprocessor()

  local srcDicts = makeDicts('source',opt.train_src)
  local tgtDicts = makeDicts('target',opt.train_tgt)

  local srcData, tgtData = preprocessor:makeBilingualData(opt.train_src,
                                                          opt.train_tgt,
                                                          srcDicts,
                                                          tgtDicts,
                                                          noFilter)

  tester:eq(torch.typename(srcData.words), 'tds.Vec')
  tester:eq(torch.typename(srcData.features), 'tds.Vec')
  tester:eq(#srcData.words, 3000)
  tester:eq(#srcData.features, 3000)

  tester:eq(torch.typename(tgtData.words), 'tds.Vec')
  tester:eq(torch.typename(tgtData.features), 'tds.Vec')
  tester:eq(#tgtData.words, 3000)
  tester:eq(#tgtData.features, 3000)
end

function preprocessorTest.monotext()
  local preprocessor, opt = buildPreprocessor('monotext')

  local dicts = makeDicts('source',opt.train)

  local data = preprocessor:makeMonolingualData(opt.train, dicts, noFilter)

  tester:eq(torch.typename(data.words), 'tds.Vec')
  tester:eq(torch.typename(data.features), 'tds.Vec')
  tester:eq(#data.words, 3000)
  tester:eq(#data.features, 3000)
end

local function isValid(seq, maxSeqLength)
  if torch.isTensor(seq) then
    return seq:size(1) > 0 and seq:size(1) <= maxSeqLength
  end
  return #seq > 0 and #seq <= maxSeqLength
end

function preprocessorTest.feattext()
  local preprocessor, opt = buildPreprocessor('feattext')

  local tgtDicts = makeDicts('target',opt.train_tgt)

  local srcData,tgtData = preprocessor:makeFeatTextData(opt.train_src,
                                                        opt.train_tgt,
                                                        tgtDicts,
                                                        isValid)

  tester:eq(torch.typename(srcData.vectors), 'tds.Vec')
  tester:eq(torch.typename(tgtData.words), 'tds.Vec')
  tester:eq(torch.typename(tgtData.features), 'tds.Vec')
  tester:eq(srcData.vectors[1]:size(2), 2)
  tester:eq(#srcData.vectors, 947)
  tester:eq(#tgtData.features, 0)
end

function preprocessorTest.tritext()
  local preprocessor, opt = buildPreprocessor('tritext')

  local src1Dicts = makeDicts('source1',opt.train_src1)
  local src2Dicts = makeDicts('source2',opt.train_src2)
  local tgtDicts = makeDicts('target',opt.train_tgt)

  local src1Data,src2Data, tgtData = preprocessor:makeTritextData(opt.train_src1,
                                                                   opt.train_src2,
                                                                   opt.train_tgt,
                                                                   src1Dicts,
                                                                   src2Dicts,
                                                                   tgtDicts,
                                                                   isValid)

  tester:eq(torch.typename(src1Data.words), 'tds.Vec')
  tester:eq(torch.typename(src2Data.words), 'tds.Vec')
  tester:eq(torch.typename(tgtData.words), 'tds.Vec')
  tester:eq(torch.typename(src1Data.features), 'tds.Vec')
  tester:eq(torch.typename(src2Data.features), 'tds.Vec')
  tester:eq(torch.typename(tgtData.features), 'tds.Vec')

  tester:eq(#src1Data.words, 3000, 200)
  tester:eq(#src1Data.features, 3000, 200)
  tester:eq(#src2Data.words, 3000, 200)
  tester:eq(#src2Data.features, 0)
  tester:eq(#tgtData.words, 3000, 200)
  tester:eq(#tgtData.features, 3000, 200)
end

return preprocessorTest
