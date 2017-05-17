require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('preprocess.lua')

-- First argument define the dataType: bitext/monotext - default is bitext.
local dataType = cmd.getArgument(arg, '-data_type') or 'bitext'

-- Options declaration
local options = {
  {
    '-data_type', 'bitext',
    [[Type of data to preprocess. Use 'monotext' for monolingual data.
      This option impacts all options choices.]],
    {
      enum = {'bitext', 'monotext', 'feattext', 'tritext'}
    }
  },
  {
    '-save_data', '',
    [[Output file for the prepared data.]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  }
}

cmd:setCmdLineOptions(options, 'Preprocess')

onmt.data.Preprocessor.declareOpts(cmd, dataType)
onmt.utils.Logger.declareOpts(cmd)

local otherOptions = {
  {
    '-seed', 3425,
    [[Random seed.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  }
}
cmd:setCmdLineOptions(otherOptions, 'Other')

local opt = cmd:parse(arg)

local function isValid(seq, maxSeqLength)
  if torch.isTensor(seq) then
    return seq:size(1) > 0 and seq:size(1) <= maxSeqLength
  end
  return #seq > 0 and #seq <= maxSeqLength
end

local function main()

  torch.manualSeed(opt.seed)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local Vocabulary = onmt.data.Vocabulary
  local Preprocessor = onmt.data.Preprocessor.new(opt, dataType)

  local data = { dataType=dataType }

  -- keep processing options in the structure for further traceability
  data.opt = opt

  data.dicts = {}

  _G.logger:info('Preparing vocabulary...')
  if dataType == 'tritext' then
    data.dicts.src = Vocabulary.init('source1',
                                     opt.train_src1,
                                     opt.src1_vocab,
                                     opt.src1_vocab_size,
                                     opt.src1_words_min_frequency,
                                     opt.features_vocabs_prefix,
                                     function(s) return isValid(s, opt.src1_seq_length) end,
                                     opt.keep_frequency,
                                     opt.idx_files)
    data.dicts.src2 = Vocabulary.init('source2',
                                     opt.train_src2,
                                     opt.src2_vocab,
                                     opt.src2_vocab_size,
                                     opt.src2_words_min_frequency,
                                     opt.features_vocabs_prefix,
                                     function(s) return isValid(s, opt.src2_seq_length) end,
                                     opt.keep_frequency,
                                     opt.idx_files)
  elseif dataType == 'bitext' or dataType == 'monotext' then
    data.dicts.src = Vocabulary.init('source',
                                     opt.train_src,
                                     opt.src_vocab,
                                     opt.src_vocab_size,
                                     opt.src_words_min_frequency,
                                     opt.features_vocabs_prefix,
                                     function(s) return isValid(s, opt.src_seq_length) end,
                                     opt.keep_frequency,
                                     opt.idx_files)
  end
  if dataType ~= 'monotext' then
    data.dicts.tgt = Vocabulary.init('target',
                                     opt.train_tgt,
                                     opt.tgt_vocab,
                                     opt.tgt_vocab_size,
                                     opt.tgt_words_min_frequency,
                                     opt.features_vocabs_prefix,
                                     function(s) return isValid(s, opt.tgt_seq_length) end,
                                     opt.keep_frequency,
                                     opt.idx_files)
  end

  _G.logger:info('Preparing training data...')
  data.train = {}
  if dataType == 'monotext' then
    data.train.src = Preprocessor:makeMonolingualData(opt.train, data.dicts.src, isValid)
  elseif dataType == 'feattext' then
    data.train.src, data.train.tgt = Preprocessor:makeFeatTextData(opt.train_src, opt.train_tgt,
                                                                   data.dicts.tgt,
                                                                   isValid)
    -- record the size of the input layer
    data.dicts.srcInputSize = data.train.src.vectors[1]:size(2)
  elseif dataType == 'tritext' then
    data.train.src, data.train.src2, data.train.tgt = Preprocessor:makeTritextData(opt.train_src1, opt.train_src2, opt.train_tgt,
                                                                    data.dicts.src, data.dicts.src2, data.dicts.tgt,
                                                                    isValid,
                                                                    Preprocessor.checkTritextUnk)
  else
    data.train.src, data.train.tgt = Preprocessor:makeBilingualData(opt.train_src, opt.train_tgt,
                                                                    data.dicts.src, data.dicts.tgt,
                                                                    isValid)
  end

  _G.logger:info('')

  _G.logger:info('Preparing validation data...')
  data.valid = {}
  if dataType == 'monotext' then
    data.valid.src = Preprocessor:makeMonolingualData(opt.valid, data.dicts.src, isValid)
  elseif dataType == 'feattext' then
    data.valid.src, data.valid.tgt = Preprocessor:makeFeatTextData(opt.valid_src, opt.valid_tgt,
                                                                    data.dicts.tgt,
                                                                    isValid)
  elseif dataType == 'tritext' then
    data.valid.src, data.valid.src2, data.valid.tgt = Preprocessor:makeTritextData(opt.valid_src1, opt.valid_src2, opt.valid_tgt,
                                                                    data.dicts.src, data.dicts.src2, data.dicts.tgt,
                                                                    isValid,
                                                                    Preprocessor.checkTritextUnk)
  else
    data.valid.src, data.valid.tgt = Preprocessor:makeBilingualData(opt.valid_src, opt.valid_tgt,
                                                                    data.dicts.src, data.dicts.tgt,
                                                                    isValid)
  end

  _G.logger:info('')

  if dataType == 'monotext' then
    if opt.vocab:len() == 0 then
      Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data)
    end
  elseif dataType == 'feattext' then
    if opt.tgt_vocab:len() == 0 then
      Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data)
    end
  elseif dataType == 'tritext' then
    if opt.src1_vocab:len() == 0 then
      Vocabulary.save('source1', data.dicts.src.words, opt.save_data .. '.src1.dict')
    end
    if opt.src2_vocab:len() == 0 then
      Vocabulary.save('source2', data.dicts.src2.words, opt.save_data .. '.src2.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      Vocabulary.saveFeatures('source1', data.dicts.src.features, opt.save_data..'.source1')
      Vocabulary.saveFeatures('source2', data.dicts.src2.features, opt.save_data..'.source2')
    end
  else
    if opt.src_vocab:len() == 0 then
      Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.src.dict')
    end
    if opt.tgt_vocab:len() == 0 then
      Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
    end
    if opt.features_vocabs_prefix:len() == 0 then
      Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data..'.source')
      Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data..'.target')
    end
  end

  _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)
  _G.logger:shutDown()
end

main()
