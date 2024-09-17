from typing import List
import numpy as np
from numpy import linalg as la

import gurobipy as gp


def oa_callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        print("Adding lazy constraint")
        # get model info
        x = model._x
        theta = model._theta
        Q = model._Q
        n = len(x)

        # Get solution
        y = np.array(model.cbGetSolution(x))
        fy = y.T @ Q @ y
        dfy = 2 * Q @ y

        # add cut
        model.cbLazy(theta <= fy + sum(dfy[i] * (x[i] - y[i]) for i in range(n)))


def oa_solve(
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
    where `K` is mixed-integer

    Aruguments:
     - Q (np.ndarray) : The matrix defining quadratic objective
     - p (np.ndarray) : The vector defining linear objective
     - x (List[gp.Var]) : The list of associated decision variables
     - mdl (gp.Model) : The model associated with variables `x`

    Returns:
     - mdl

    Raises:
     - Exception : If for whatever reason the model does not solve
    """
    n = len(p)
    # Add epigraph values and objective
    Q_norm = float(la.norm(Q) ** 2)
    theta = mdl.addVar(lb=-Q_norm, ub=Q_norm, name="epigraph")
    mdl.setObjective(theta + sum(p[i] * x[i] for i in range(n)), gp.GRB.MAXIMIZE)

    # add info to model
    mdl._x = x
    mdl._theta = theta
    mdl._Q = Q

    # solve with callback and return
    mdl.Params.LazyConstraints = 1
    mdl.optimize(oa_callback)

    if mdl.status != gp.GRB.OPTIMAL:
        # TODO: Add better exception handelling
        raise Exception("Something went wrong, model did not solve")

    return mdl
