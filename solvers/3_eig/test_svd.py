import numpy as np
import eigenthings as et

A = np.random.rand(40,10)

print('np.linalg.svd(A)...',end='')
U, S, Vh = np.linalg.svd(A)
print('Done')

print('np.linalg.eigh(AtA)...',end='')
AtA = A.T @ A
l, mV = np.linalg.eigh(AtA)
indexes = np.argsort(-l)
l = l[indexes]
mV = mV[:,indexes]
mV /= np.linalg.norm(mV,axis=0)
mS = np.zeros(A.shape[0],dtype=A.dtype)
mS[:l.size] = np.sqrt(l[:])
mU = np.zeros((A.shape[0],A.shape[0]),dtype=A.dtype)
sz = np.sum(mS>0)
mU[:,:sz] = (A @ mV) * (1/mS[:sz])
mU, _ = np.linalg.qr(mU)
print('Done')
print(f'S error: ',np.max(np.abs(mS[:l.size]-S)))
print(f'U error: ',np.max( np.linalg.norm(np.abs(mU[:,:sz])-np.abs(U[:,:sz]),axis=0) ) )
print(f'V error: ',np.max( np.linalg.norm(np.abs(mV)-np.abs(Vh.T),axis=0) ) )

print('et.svd(A)...',end='')
mS, mU, mV = et.svd(A,tol=1e-08)
sz = np.sum(mS>0)
print('Done')
print(f'S error: ',np.max(np.abs(mS[:l.size]-S)))
print(f'U error: ',np.max( np.linalg.norm(np.abs(mU[:,:sz])-np.abs(U[:,:sz]),axis=0) ) )
print(f'V error: ',np.max( np.linalg.norm(np.abs(mV)-np.abs(Vh.T),axis=0) ) )

