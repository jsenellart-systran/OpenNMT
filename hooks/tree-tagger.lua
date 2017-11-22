local path = require('pl.path')

local hc = require('httpclient').new()

local myopt =
{
  {
    '-pos_feature', '',
    [[Use treetagger to inject pos tags, the parameter is the path to the model to use. `treetagger`
      is expected to be in found in executable path.]],
    {
      valid = function(v) return v == '' or path.exists(v), 'the file must exist' end
    }
  },
  {
    '-pos_server_host', 'localhost',
    [[POS server to use.]]
  },
  {
    '-pos_server_port', 3000,
    [[Port on the POS server to use.]]
  }
}

local function declareOptsFn(cmd)
  cmd:setCmdLineOptions(myopt, 'Tokenizer')
end

local function treetaggerFn(opt, tokens)
  local tok_nofeats = ''
  for _,v in ipairs(tokens) do
    local p = v:find('￨')
    if p then
      v = v:sub(1,p)
    end
    if tok_nofeats ~= '' then
      tok_nofeats = tok_nofeats..' '
    end
    tok_nofeats = tok_nofeats..v
  end
  local res = hc:post("http://"..opt.pos_server_host..':'..opt.pos_server_port..'/pos', 'sent='..tok_nofeats)
  assert(res.code==200)
  local idx = 1
  for pos in string.gmatch(res.body, "%S+") do
    tokens[idx] = tokens[idx] .. '￨' .. pos
    idx = idx + 1
  end
  return tokens
end

return {
  post_tokenize = treetaggerFn,
  declareOpts = declareOptsFn
}
