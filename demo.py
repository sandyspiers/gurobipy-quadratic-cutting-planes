import numpy as np

import gurobipy as gp

from cutting_planes import ecp_solve, oa_solve, quad_solve

# # Solve a random boolean quadratic problem, using continuous and binary vars
# Generate problem
n = 10
r = 2
m = 5

Q = np.random.randint(0, 100, size=(n, r))
Q = -(Q @ Q.T)
p = np.random.randint(0, 100, size=n)

# # Demo extended cutting plane on continuous model
mdl = gp.Model()
x = [mdl.addVar(ub=1) for _ in range(n)]
mdl.addConstr(sum(x) == m)
mdl.setParam("OutputFlag", 0)

quad = quad_solve(Q, p, x, mdl).ObjVal
ecp = ecp_solve(Q, p, x, mdl).ObjVal
print(f"ECP:{ecp}\tQUAD:{quad}")

# # Demo outer approximation on binary model
mdl = gp.Model()
x = [mdl.addVar(vtype=gp.GRB.BINARY) for _ in range(n)]
mdl.addConstr(sum(x) == m)

quad = quad_solve(Q, p, x, mdl).ObjVal
oa = oa_solve(Q, p, x, mdl).ObjVal
print(f"OA:{oa}\tQUAD:{quad}")
