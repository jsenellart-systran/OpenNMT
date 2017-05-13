require('onmt.init')

local tester = ...

local manhattanDistanceTest = torch.TestSuite()

function manhattanDistanceTest.basic()
  local t=torch.Tensor(3,10):uniform(0.1)
  local u=torch.Tensor(3,10):uniform(0.1)
  local d = onmt.ManhattanDistance()
  for i = 1, 3 do
    tester:eq(d:forward({t,u})[i], (t[i]-u[i]):abs():sum())
  end
end

function manhattanDistanceTest.expNorm()
  local t=torch.Tensor(3,10):uniform(0.1)
  local u=torch.Tensor(3,10):uniform(0.1)
  local d = onmt.ManhattanDistance(1)
  for i = 1, 3 do
    tester:assertge(d:forward({t,u})[i], 0)
    tester:assertlt(d:forward({t,u})[i], 1)
  end
end

return manhattanDistanceTest
