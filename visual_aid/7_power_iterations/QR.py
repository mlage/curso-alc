import numpy as np

def standard(A,tol=1e-12):
    m,n = A.shape
    if n<m:
      return factorize_square_R(A)
    Q = np.zeros((m,m),dtype=A.dtype)
    R = np.zeros_like(A)
    u = np.zeros((1,m),dtype=A.dtype)
    dots = np.zeros(m,dtype=A.dtype)
    for i in range(m):
        u[0,:] = A[:,i]
        dots[:i] = u @ Q[:,:i] 
        u[0,:] -= Q[:,:i] @ dots[:i]  
        delta = np.linalg.norm(u)
        if delta>1e-12:#tol:
          u /= delta
        else:
          u[0,:] = 0.
        Q[:,i] = u[0,:]
        R[:i,i] = dots[:i]
        R[i,i] = delta
    for i in range(m,n):
        u[0,:] = A[:,i]
        dots[:] = u @ Q
        R[:,i] = dots[:]
    return Q, R

def factorize_square_R(A):
    m,n = A.shape
    Q = np.zeros_like(A)
    R = np.zeros((n,n),dtype=A.dtype)
    u = np.zeros((1,m),dtype=A.dtype)
    dots = np.zeros(n,dtype=A.dtype)
    for i in range(n):
        u[0,:] = A[:,i]
        dots[:i] = u @ Q[:,:i] 
        u[0,:] -= Q[:,:i] @ dots[:i]  
        delta = np.linalg.norm(u)
        if delta>1e-14:
          u /= delta
        else:
          u[0,:] = 0.
        Q[:,i] = u[0,:]
        R[:i,i] = dots[:i]
        R[i,i] = delta
    return Q, R

def factorize_square_Q(A):
    m,n = A.shape
    Q = np.zeros((m,m),dtype=A.dtype)
    R = np.zeros_like(A)
    P = np.arange(n)
    u = np.zeros((1,m),dtype=A.dtype)
    dots = np.zeros(m,dtype=A.dtype)
    end_col = n-1
    n_zeros = 0
    for i in range(m):
        swapped_cols = True
        while swapped_cols:
          u[0,:] = A[:,i]
          dots[:i] = u @ Q[:,:i] 
          u[0,:] -= Q[:,:i] @ dots[:i]  
          delta = np.linalg.norm(u)
          if delta>1e-14:
            u /= delta
            swapped_cols = False
          else:
            if end_col <= i:
              return Q, R, P
            u[0,:] = 0.
            n_zeros += 1
            P[i], P[end_col] = P[end_col], P[i]
            end_col = end_col-1
        Q[:,i] = u[0,:]
        R[:i,i] = dots[:i]
        R[i,i] = delta
    return Q, R, P, n_zeros










