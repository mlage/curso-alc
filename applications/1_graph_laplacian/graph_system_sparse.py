import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse

#import LU
#import steepest_descent
import conjugate_gradient

def build_sparse_laplacian_matrix_for_grid(rows,cols):
    nodes = rows*cols
    
    I = np.zeros(nodes*5,dtype='uint32')
    J = np.zeros(nodes*5,dtype='uint32')
    V = np.zeros(nodes*5,dtype='float32')

    # diagonal terms
    I[:nodes] = np.arange(nodes)
    J[:nodes] = I[:nodes]
    V[:nodes] = 4.
    V[0] = 2.
    V[cols-1] = 2.
    V[nodes-cols] = 2.
    V[nodes-1] = 2.
    V[1:cols-1]=3.
    V[nodes-cols+1:nodes-1]=3.
    V[cols:nodes-cols:cols]=3.
    V[2*cols-1:nodes-1:cols]=3.

    # horizontal off-diagonal terms
    ii = np.arange(1,nodes)
    jj = np.arange(0,nodes-1)
    I[nodes:nodes+len(ii)] = ii
    J[nodes:nodes+len(jj)] = jj
    I[nodes+len(ii):nodes+2*len(ii)] = jj
    J[nodes+len(jj):nodes+2*len(jj)] = ii
    V[nodes:nodes+2*len(ii)] = -1.
    V[nodes+cols-1:nodes+len(ii):cols] = 0.
    V[nodes+len(ii)+cols-1:nodes+2*len(ii):cols] = 0.

    # vertical off-diagonal terms
    aux = nodes+2*len(ii)
    ii = np.arange(cols,nodes)
    jj = np.arange(0,nodes-cols)
    I[aux:aux+len(ii)] = ii
    J[aux:aux+len(jj)] = jj
    I[aux+len(ii):aux+2*len(ii)] = jj
    J[aux+len(jj):aux+2*len(jj)] = ii
    V[aux:aux+2*len(ii)] = -1.

    # build sparse matrix from (i,j,v) triplets
    A = sparse.csr_matrix((V,(I,J)),dtype='float32')
    
    return A
  
#######################################################

if __name__ == '__main__':

    rows = 100
    cols = 100
    nodes = rows*cols

    A = build_sparse_laplacian_matrix_for_grid(rows,cols)

    #boundary conditions
    A.data[:7+4*(cols-2)]=0.
    A.data[-7-4*(cols-2):]=0.
    A[np.arange(cols),np.arange(cols)]=1.
    A[np.arange(nodes-cols,nodes),np.arange(nodes-cols,nodes)]=1.

    b = np.zeros((rows-2)*cols)

    b[:] = (-np.sum(A[cols:(rows-1)*cols,:cols]*2.0,axis=1)-np.sum(A[cols:(rows-1)*cols,-cols:]*2.0,axis=1)).flatten()

    b[cols*(rows//2-rows//10):cols*(rows//2+rows//10):cols] = -0.1

    x_CG = conjugate_gradient.solve(A[cols:nodes-cols,cols:nodes-cols],b,tol=1e-04,maxit=2000)

    plt.figure(1)
    plt.imshow(x_CG.reshape(rows-2,cols),cmap='jet')
    plt.colorbar()
    plt.show()
    
    img = np.zeros((rows,cols),dtype='float32')
    img[0,:] = 2.
    img[-1,:] = 2.
    img[1:-1,:] = x_CG.reshape(rows-2,cols)
    img.tofile(f'potential_{rows}x{cols}.bin')
