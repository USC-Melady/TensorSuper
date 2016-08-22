import numpy as np
import sys
sys.path.append('/afs/cs.stanford.edu/u/yuqirose/cvxpy')
from cvxpy import *
sys.path.append('/afs/cs.stanford.edu/u/yuqirose/CVXcanon')
from CVXcanon import *


var_shape = (2,12)
mode = 1
X_n = np.random.random((2,12))
Omega_n = np.ones((2,12))    


X_opt = Variable(var_shape[0], var_shape[1])
X_opt_n = X_opt
#X_opt_n = unfold(X_opt, var_shape, mode)


obj = Minimize(norm(X_opt_n))
constraints = [mul_elemwise(Omega_n, X_opt_n) == mul_elemwise(Omega_n, X_n)]
prob = Problem(obj, constraints)
# Use SCS to solve the problem.
prob.solve(verbose=True, solver=SCS)


