import numpy as np

import LU
import steepest_descent

adjacency = np.array([ [0,1,1,0,0],
                       [1,0,0,0,1],
                       [1,0,0,1,0],
                       [0,0,1,0,1],
                       [0,1,0,1,0] ],dtype='float64')

degrees = np.sum(adjacency,axis=1)
                       
laplacian = np.diag(degrees) - adjacency

print(laplacian)

# boundary conditions
x_0 = 2.
x_1 = 1.
A = laplacian[2:,2:]
b = -laplacian[2:,0]*x_0-laplacian[2:,1]*x_1

x_LU = LU.solve(A,b)
x_SD = steepest_descent.solve_symmetric(A,b)

print("x_LU =",x_LU)
print("x_SD =",x_SD)
