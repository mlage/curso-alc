'''
ATENÇÃO:

Este código foi feito apenas com intuito educacional,
para demonstrar algumas variações do algoritmo QR para
encontrar autovalores e autovetores de matrizes simétricas reais.

Não utilize estas funções para matrizes grandes, terá problemas.

1) Eficiência não foi uma preocupação aqui.
2) Faltam componentes importantes para solução robusta da eigendecomposition
   de matrizes grandes:
   
   a) Nenhuma estratégia de deflação foi implementada.
   b) Os shifts usados aqui são bastante simples. Podem ser melhorados.
      Aos interessados, sugere-se pesquisar sobre shifts de Wilkinson.
'''

import numpy as np

#######################################
def svd(M,maxit=1000,tol=1e-06):
    A = M.T @ M if M.shape[0] >= M.shape[1] else M @ M.T
    L, V, _, _ = QR_hessenberg(A,maxit=maxit,tol=tol)
    indexes = np.argsort(-L)
    L = L[indexes]
    V = V[:,indexes]
    sigmas = np.zeros(max(M.shape),dtype=M.dtype)
    sigmas[:L.size] = np.sqrt(L[:])
    U = np.zeros((M.shape[0],M.shape[0]),dtype=M.dtype)
    sz = np.sum(sigmas>0)
    if M.shape[0] >= M.shape[1]:
      U[:,:sz] = (M @ V) * (1/sigmas[:sz])
      U, _ = np.linalg.qr(U)
    else:
      U = V.copy()
      V[:,:sz] = (M.T @ V) * (1/sigmas[:sz])
      V, _ = np.linalg.qr(V)
    return sigmas, U, V

#######################################
def QR_simple(M,maxit=1000,tol=1e-06):
    A = M.copy()
    ii, jj = subdiagonal_indexes(A.shape[0])
    deltas = [ np.max(np.abs(A[ii,jj])) ]
    V = np.eye(A.shape[0])
    it = 0
    while it < maxit and deltas[-1] > tol:
        it = it + 1
        Q, R = np.linalg.qr(A)#QR.standard(A,tol)
        A = R @ Q
        V = V @ Q
        deltas.append(np.max(np.abs(A[ii,jj])))
    if deltas[-1] < tol:
      print(f'QR algorithm converged within {it} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    else:
      print(f'WARNING: QR algorithm did not converge within {maxit} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    return np.array(np.diag(A)), V, np.array(deltas), A
    
#######################################
    
def QR_with_shifts(M,maxit=1000,tol=1e-06):
    A = M.copy()
    ii, jj = subdiagonal_indexes(A.shape[0])
    deltas = [ np.max(np.abs(A[ii,jj])) ]
    V = np.eye(A.shape[0])
    diag_ones = np.diag(np.ones(A.shape[0]))
    it = 0
    while it < maxit and deltas[-1] > tol:
        it = it + 1
        s = A[-1,-1] if np.max( np.abs(A[-1,:-1]) ) > tol else tol
        Q, R = np.linalg.qr(A - s*diag_ones)#QR.standard( A - s*diag_ones,tol )
        A = R @ Q + s*diag_ones
        V = V @ Q
        deltas.append(np.max(np.abs(A[ii,jj])))
    if deltas[-1] < tol:
      print(f'QR algorithm converged within {it} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    else:
      print(f'WARNING: QR algorithm did not converge within {maxit} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    return np.array(np.diag(A)), V, np.array(deltas), A
    
#######################################
    
def QR_hessenberg(M,maxit=1000,tol=1e-06):
    A = M.copy()
    V = np.eye(A.shape[0])
    for i in range(A.shape[0]-2):
        householder_left(A,i,hessenberg=True,V=V)
    ii, jj = subdiagonal_indexes(A.shape[0])
    deltas = [ np.max(np.abs(A[ii,jj])) ]
    diag_ones = np.diag(np.ones(A.shape[0]))
    it = 0
    while it < maxit and deltas[-1] > tol:
        it = it + 1
        Q, R = np.linalg.qr(A)#QR.standard(A,tol)
        A = R @ Q
        V = V @ Q
        deltas.append(np.max(np.abs(A[ii,jj])))
    if deltas[-1] < tol:
      print(f'QR algorithm converged within {it} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    else:
      print(f'WARNING: QR algorithm did not converge within {maxit} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    return np.array(np.diag(A)), V, np.array(deltas), A
    
#######################################
    
def QR_with_shifts_and_hessenberg(M,maxit=1000,tol=1e-06):
    A = M.copy()
    V = np.eye(A.shape[0])
    for i in range(A.shape[0]-2):
        householder_left(A,i,hessenberg=True,V=V)
    ii, jj = subdiagonal_indexes(A.shape[0])
    deltas = [ np.max(np.abs(A[ii,jj])) ]
    diag_ones = np.diag(np.ones(A.shape[0]))
    it = 0
    while it < maxit and deltas[-1] > tol:
        it = it + 1
        s = A[-1,-1] if np.max( np.abs(A[-1,:-1]) ) > tol else tol
        Q, R = np.linalg.qr(A - s*diag_ones)#QR.standard( A - s*diag_ones,tol )
        A = R @ Q + s*diag_ones
        V = V @ Q
        deltas.append(np.max(np.abs(A[ii,jj])))
    if deltas[-1] < tol:
      print(f'QR algorithm converged within {it} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    else:
      print(f'WARNING: QR algorithm did not converge within {maxit} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    return np.array(np.diag(A)), V, np.array(deltas), A
    
#######################################
    
def QR_hessenberg_and_givens(M,maxit=1000,tol=1e-06):
    A = M.copy()
    V = np.eye(A.shape[0])
    for i in range(A.shape[0]-2):
        householder_left(A,i,hessenberg=True,V=V)
    ii, jj = subdiagonal_indexes(A.shape[0])
    deltas = [ np.max(np.abs(A[ii,jj])) ]
    diag_ones = np.diag(np.ones(A.shape[0]))
    it = 0
    while it < maxit and deltas[-1] > tol:
        it = it + 1
        for k in range(A.shape[0]-1):
            givens_rotation_left(A,k,tol=tol,similarity=True,V=V)
        deltas.append(np.max(np.abs(A[ii,jj])))
    if deltas[-1] < tol:
      print(f'QR algorithm converged within {it} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    else:
      print(f'WARNING: QR algorithm did not converge within {maxit} steps. delta = {deltas[-1]:.2e}, tol = {tol:.2e}')
    return np.array(np.diag(A)), V, np.array(deltas), A
    
#######################################
    
def hessenberg(A):
    n = A.shape[0]
    B = A.copy()
    for i in range(n-2):
        householder_left(B,i,hessenberg=True)
    return B

#######################################

def givens_rotation_left(A,i,tol=1e-06,similarity=False,V=None):
  row = i+1
  if np.abs(A[row,i]) < tol:
    return
  a = A[i,i]
  b = A[row,i]
  t = a/b
  s = -1. / np.sqrt(t**2+1)
  c = -t * s
  R = np.array([[c,-s],[s,c]])
  A[[i,row],:] = R @ A[[i,row],:]
  if similarity:
    A[:,[i,row]] = A[:,[i,row]] @ R.T
    if V is not None:
        V[:,[i,row]] = V[:,[i,row]] @ R.T

def householder_left(A,i,hessenberg=False,V=None):
  row = i if not hessenberg else i+1
  x = A[row:,i]
  e1 = np.zeros_like(x)
  e1[0] = 1
  v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
  beta = 2 / np.dot(v,v)
  q = A[row:,:].T @ v
  A[row:,:] -= np.einsum( 'i,j->ij' ,beta * v, q )
  if hessenberg:
    q = A[:,row:] @ v
    A[:,row:] -= np.einsum( 'i,j->ij' ,beta * q, v )
    if V is not None:
      q = V[:,row:] @ v
      V[:,row:] -= np.einsum( 'i,j->ij' ,beta * q, v )
  
def householder_right(A,i,hessenberg=False):
  col = i if not hessenberg else i+1
  x = A[i,col:]
  e1 = np.zeros_like(x)
  e1[0] = 1
  v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
  beta = 2 / np.dot(v,v)
  q = A[i:,col:] @ v
  A[i:,col:] -= np.einsum( 'i,j->ij' ,beta * q, v )
  if hessenberg:
    q = A[col:,i:].T @ v
    A[col:,i:] -= np.einsum( 'i,j->ij' ,beta * v, q )
  
def subdiagonal_indexes(n):
  sz = (n*(n-1))//2
  i = np.zeros(sz,dtype='uint32')
  j = np.zeros(sz,dtype='uint32')
  i[:n-1] = np.arange(1,n)
  j[:n-1] = 0
  offset = n-1
  for k in range(1,n):
    i[offset:offset+(n-1-k)] = i[offset-(n-1-k):offset] #np.arange(k+1,n)
    j[offset:offset+(n-1-k)] = k
    offset += n-1-k
  return i,j
