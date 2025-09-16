import numpy as np

n=64
file_name = f"wavelet_basis_{n}"
file_ext = ".bin"
A = np.zeros((n,n),dtype='float32')

col=1
A[:,0]=1
while n>1:
    n = n//2
    A[:n,col]=1
    A[n:2*n,col]=-1
    for i in range(1,col):
        A[2*n*i:2*n*i+n,col+i]=1
        A[2*n*i+n:2*n*(i+1),col+i]=-1
    col *= 2
    #print(A)

iA = np.linalg.inv(A)
A.tofile(file_name+file_ext)
iA.tofile(file_name+"_inv"+file_ext)

