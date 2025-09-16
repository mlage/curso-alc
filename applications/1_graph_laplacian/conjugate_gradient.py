import numpy as np

# works for symmetric positive-definite matrices
def solve(A,b,x=None,tol=1e-06,maxit=1000):
    delta_0 = np.dot(b,b)
    if delta_0 <= tol*tol:
      print(f"Null vector satisfied Ax=b within provided tolerance {tol:.4e}.")
      return np.zeros_like(b)

    x = x if x is not None else np.zeros_like(b)
    r = np.zeros_like(b)
    d = np.zeros_like(b)
    q = np.zeros_like(b)
    
    r[:] = b - A @ x
    delta = np.dot(r,r)
    beta = 0.
    
    it = 0
    rel_tol = delta_0*tol*tol
    while delta > rel_tol and it < maxit:
      it = it + 1
      d[:] = beta*d + r
      q[:] = A @ d
      alpha = delta / np.dot(q,d)
      x[:] = x + alpha*d
      r[:] = r - alpha*q
      delta_prev = delta
      delta = np.dot(r,r)
      beta = delta/delta_prev
    print(f"Solver stopped at iteration {it}, with residual {np.sqrt(delta/delta_0):.4e}")
    
    return x
