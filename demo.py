import numpy as np

import gurobipy as gp

from cutting_planes import ecp_solve

n = 10
r = 2
m = 5

Q = np.random.randint(0, 100, size=(n, r))
Q = -(Q @ Q.T)
p = np.random.randint(0, 100, size=n)

print(np.linalg.eig(Q))


mdl = gp.Model()
x = [mdl.addVar(ub=1) for _ in range(n)]
mdl.addConstr(sum(x) == m)
mdl.setParam("OutputFlag", 0)

ecp_solve(Q, p, x, mdl)
