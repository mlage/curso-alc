import numpy as np
import eigenthings as et
from matplotlib import pyplot as plt

A = np.random.rand(10,10)
A = A.T @ A

print('np.linalg.eigh(A)...',end='')
l, v = np.linalg.eigh(A)
print('Done')

indexes = np.argsort(l)
l = l[indexes]
v = v[:,indexes]
v /= np.linalg.norm(v,axis=0)

labels = [ 'QR Simple', 'QR with shifts', 'QR Hessenberg', 'QR with shifts Hessenberg', 'QR Hessenberg Givens' ]
funcs = [ et.QR_simple, et.QR_with_shifts, et.QR_hessenberg, et.QR_with_shifts_and_hessenberg, et.QR_hessenberg_and_givens ]
deltas = []

for label, my_eig_func in zip(labels,funcs):
  print(f'{label}...')
  my_l, my_v, my_deltas, _ = my_eig_func(A,maxit=10_000,tol=1e-08)
  print('Done')
  indexes = np.argsort(my_l)
  my_l = my_l[indexes]
  my_v = my_v[:,indexes]
  my_v /= np.linalg.norm(my_v,axis=0)
  print(f'Eigenvals error: ',np.max(np.abs(l-my_l)))
  print(f'Eigenvecs error: ',np.max( np.linalg.norm(np.abs(v)-np.abs(my_v),axis=0) ) )
  deltas.append(my_deltas)

for delta in deltas:
  plt.plot( np.arange(delta.size), delta )
plt.legend( labels )
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Residual')

plt.show()

