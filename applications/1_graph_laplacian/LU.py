import numpy as np

def swap_rows(A, i, j, partial=False, start=None, end=None):
    if not partial:
        A[[i, j],:] = A[[j, i],:]
        return
    start = i if start is None else start
    end = A.shape[1] if end is None else end
    A[[i, j],start:end] = A[[j, i],start:end]

def swap_cols(A, i, j, partial=False, start=None, end=None):
    if not partial:
        A[:, [i, j]] = A[:, [j, i]]
        return
    start = i if start is None else start
    end = A.shape[0] if end is None else end
    A[start:end, [i, j]] = A[start:end, [j, i]]
    
def permutate_forward(A,row=None,col=None):
    if row is None and col is None:
        return A
    if len(A.shape)==1:
      ids = row if row is not None else col
      x = np.array(A[ids])
      return x
    M = np.array(A)
    if col is None:
        M = A[row,:]
    elif row is None:
        M = A[:,col]
    else:
        M = A[np.ix_(row,col)]
    return M

def permutate_backward(A,row=None,col=None):
    if row is None and col is None:
        return A
    if len(A.shape)==1:
      ids = row if row is not None else col
      ids_inv = np.zeros_like(ids)
      ids_inv[ids] = np.arange(len(ids))
      x = np.array(A[ids_inv])
      return x
    M = np.array(A)
    if col is None:
        row_inv = np.zeros_like(row)
        row_inv[row] = np.arange(len(row))
        M = A[row_inv,:]
    elif row is None:
        col_inv = np.zeros_like(col)
        col_inv[col] = np.arange(len(col))
        M = A[:,col_inv]
    else:
        row_inv = np.zeros_like(row)
        row_inv[row] = np.arange(len(row))
        col_inv = np.zeros_like(col)
        col_inv[col] = np.arange(len(col))
        M = A[np.ix_(row_inv,col_inv)]
    return M
         

def PQLU(A, tol=1e-15):

    # P A Q = LU
    
    n_rows, n_cols = A.shape
    U = np.array(A, dtype='float64') # copy
    L = np.eye(n_rows)
    P = np.arange(n_rows)
    Q = np.arange(n_cols)
    
    for i in range(n_rows-1):
    
      pivot = np.argmax( np.abs(U[i:,i]) ) + i
      if pivot != i:
        swap_rows(U,i,pivot,partial=True)
        swap_rows(L,i,pivot,partial=True,start=0,end=i)
        P[i], P[pivot] = P[pivot], P[i]
      elif np.abs(U[i,i])<tol:
        pivot = np.argmax( np.abs(U[i,i:]) ) + i
        if np.abs(U[i,pivot])<tol:
            return L, U, P, Q
        swap_cols(U,i,pivot)
        Q[i], Q[pivot] = Q[pivot], Q[i]
    
      L[i+1:,i] = U[i+1:,i] / U[i,i]
      U[i+1:,i:] -= np.einsum('i,j->ij', L[i+1:,i] , U[i,i:] )
            
    return L, U, P, Q

def PQLU_full_pivot(A, tol=1e-15):

    # P A Q = LU
    
    n_rows, n_cols = A.shape
    U = np.array(A, dtype='float64') # copy
    L = np.eye(n_rows)
    P = np.arange(n_rows)
    Q = np.arange(n_cols)
    
    for i in range(n_rows-1):
    
      # find max possible pivot in all of U[i:,i:] (submatrix)
      pivot = np.argmax( np.abs(U[i:,i:]) )
      pivot_row, pivot_col = pivot // (n_cols-i), pivot % (n_cols-i)
      pivot_row += i
      pivot_col += i
      
      # Swap rows if needed
      if pivot_row != i:
          swap_rows(U, i, pivot_row, partial=True)
          swap_rows(L, i, pivot_row, partial=True, start=0, end=i)
          P[i], P[pivot_row] = P[pivot_row], P[i]
      
      # Swap columns if needed
      if pivot_col != i:
          swap_cols(U, i, pivot_col)
          Q[i], Q[pivot_col] = Q[pivot_col], Q[i]
      
      # Early termination if matrix is not full-rank
      if np.abs(U[i,i]) < tol:
          return L, U, P, Q
    
      L[i+1:,i] = U[i+1:,i] / U[i,i]
      U[i+1:,i:] -= np.einsum('i,j->ij', L[i+1:,i] , U[i,i:] )
            
    return L, U, P, Q
    
def solve_lower_triangle(L,b):
    x = np.zeros_like(b)
    for i in range(len(b)):
        x[i] = (b[i] - np.dot(L[i,:i],x[:i]))/L[i,i]
    return x
    
def solve_upper_triangle(U,b):
    x = np.zeros_like(b)
    for i in range(len(b)-1,-1,-1):
        x[i] = (b[i] - np.dot(U[i,i+1:],x[i+1:]))/U[i,i]
    return x
    
def solve(A,b):
    L, U, P, Q = PQLU_full_pivot(A)
    y = solve_lower_triangle(L,permutate_forward(b,P))
    return permutate_backward(solve_upper_triangle(U,y),Q)

if __name__ == '__main__':
    
    '''A = np.array( [ [1,0,2,1],
                    [1,3,-1,3],
                    [0,1,-1,2],
                    [2,0,4,-2] ], dtype='float64' )'''
    
    '''m, n = 100, 200
    ids = np.unique( np.random.randint(0,n,n//3) )
    cols = [ i for i in range(n) if i not in ids ]           
    A = np.random.rand(m,n)*4.-2. # all values within [-2,2]
    A[:,ids] = A[:,cols] @ (np.random.rand(len(cols),len(ids))*2.-1.) # random linear combinations
    '''
    m, n = 20, 20        
    A = np.random.rand(m,n)*4.-2. # all values within [-2,2]
    x = np.arange(1,n+1,dtype='float64')
    b = A @ x
    
    x_solved = solve(A,b)
    
    print("(original) x =",x)
    print("( solved ) x =",x_solved)
