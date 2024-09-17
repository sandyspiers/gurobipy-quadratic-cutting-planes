from typing import List
import numpy as np
from numpy import linalg as la

import gurobipy as gp


def quad_solve(
    Q: np.ndarray,
    p: np.ndarray,
    x: List[gp.Var],
    mdl: gp.Model,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
    MAX_ITERATION=999,
):
    """
    Use Gurobi quadratic programming techniques to solve
    ```
    max <Qx,x> + <p,x>
    s.t. x in K
    ```
    where `K` is either continuous or discrete

    Aruguments:
     - Q (np.ndarray) : The matrix defining quadratic objective
     - p (np.ndarray) : The vector defining linear objective
     - x (List[gp.Var]) : The list of associated decision variables
     - mdl (gp.Model) : The model associated with variables `x`

    Returns:
     - mdl

    Raises:
     - Exception : If for whatever reason the model does not solve.
    """
    n = len(x)
    mdl.setObjective(
        sum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
        + sum(p[i] * x[i] for i in range(n)),
        gp.GRB.MAXIMIZE,
    )

    # solve and return
    mdl.optimize()

    if mdl.status != gp.GRB.OPTIMAL:
        # TODO: Add better exception handelling
        raise Exception("Something went wrong, model did not solve")

    return mdl
