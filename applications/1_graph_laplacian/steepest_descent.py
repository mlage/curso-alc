import numpy as np

def solve(A,b,x=None,tol=1e-06,maxit=1000):
    delta_0 = np.dot(b,b)
    if delta_0 <= tol*tol:
      print(f"Null vector satisfied Ax=b within provided tolerance {tol:.4e}.")
      return np.zeros_like(b)

    x = x if x is not None else np.zeros_like(b)
    r = np.zeros_like(b)
    d = np.zeros_like(b)
    q = np.zeros_like(b)
    s = np.zeros_like(b)
    
    r[:] = b - A @ x
    delta = np.dot(r,r)
    
    it = 0
    rel_tol = delta_0*tol*tol
    while delta > rel_tol and it < maxit:
      it = it + 1
      d[:] = A.T @ r
      q[:] = A @ d
      s[:] = A.T @ q
      alpha = np.dot(d,d) / np.dot(s,d)
      x[:] = x + alpha*d
      r[:] = r - alpha*q # r = b - Ax
      delta = np.dot(r,r)
    print(f"Solver stopped at iteration {it}, with residual {np.sqrt(delta/delta_0):.4e}")
    
    return x

def solve_symmetric(A,b,x=None,tol=1e-06,maxit=1000):
    delta_0 = np.dot(b,b)
    if delta_0 <= tol*tol:
      print(f"Null vector satisfied Ax=b within provided tolerance {tol:.4e}.")
      return np.zeros_like(b)

    x = x if x is not None else np.zeros_like(b)
    r = np.zeros_like(b)
    q = np.zeros_like(b)
    
    r[:] = b - A @ x
    delta = np.dot(r,r)
    
    it = 0
    rel_tol = delta_0*tol*tol
    while delta > rel_tol and it < maxit:
      it = it + 1
      q[:] = A @ r
      alpha = delta / np.dot(q,r)
      x[:] = x + alpha*r
      r[:] = r - alpha*q
      delta = np.dot(r,r)
    print(f"Solver stopped at iteration {it}, with residual {np.sqrt(delta/delta_0):.4e}")
    
    return x
