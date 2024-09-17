from typing import List
import numpy as np
from numpy import linalg as la

import gurobipy as gp


def ecp_solve(
    Q: np.ndarray,
    p: np.ndarray,
    x: List[gp.Var],
    mdl: gp.Model,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
    MAX_ITERATION=999,
):
    """
    Use extended cutting planes to solve
    ```
    max <Qx,x> + <p,x>
    s.t. x in K
    ```

    Aruguments:
     - Q (np.ndarray) : The matrix defining quadratic objective
     - p (np.ndarray) : The vector defining linear objective
     - x (List[gp.Var]) : The list of associated decision variables
     - mdl (gp.Model) : The model associated with variables `x`

    Returns:
     - mdl

    Raises:
     - Exception : If for whatever reason the model does not solve at *any* step.
    """
    n = len(p)
    # Add epigraph values and objective
    Q_norm = float(la.norm(Q) ** 2)
    theta = mdl.addVar(lb=-Q_norm, ub=Q_norm, name="epigraph")
    mdl.setObjective(theta + sum(p[i] * x[i] for i in range(n)), gp.GRB.MAXIMIZE)

    # start iterations
    LB = -1e99
    UB = 1e99
    gap = 100
    iteration = 1
    while gap > rel_tol and UB - LB >= abs_tol and iteration < MAX_ITERATION:
        mdl.optimize()
        if mdl.status != gp.GRB.OPTIMAL:
            # TODO: Add better exception handelling
            raise Exception("Something went wrong, model did not solve")

        # get solution
        y = np.array([_x.X for _x in x])
        fy = y.T @ Q @ y / 2
        dfy = Q @ y

        # update bounds
        LB = max(LB, float(fy + p @ y))
        UB = mdl.ObjVal
        gap = (UB - LB) / abs(LB)

        # add cut
        mdl.addConstr(theta <= fy + sum(dfy[i] * (x[i] - y[i]) for i in range(n)))

        print(f"{LB:>18.4f}{UB:>18.4f}{gap:>8.2f}")

        iteration += 1
    return mdl
