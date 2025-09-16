import numpy as np
from matplotlib import pyplot as plt

A = np.array( [ np.random.rand(2)*2.-1.,np.random.rand(2)*2.-1. ] )
while np.linalg.matrix_rank(A)<2:
    A = np.array( [ np.random.rand(2)*2.-1.,np.random.rand(2)*2.-1. ] )

n = 1000
x = np.zeros(n,dtype='float64')
y = np.zeros(n,dtype='float64')
c = np.zeros(n,dtype='float64')
for i in range(n):
    v = np.random.rand(2)*2.-1.
    x[i] = v[0]
    y[i] = v[1]
    z = A @ v
    c[i] = np.dot(z,z)

Xm, Ym = np.meshgrid(np.linspace(np.min(x),np.max(x),100), np.linspace(np.min(y),np.max(y),100))
xy = np.stack([Xm.ravel(), Ym.ravel()], axis=1)
AtA = A.T @ A
z = np.einsum('ij,ij->i', xy @ AtA, xy)  # vectorized quadratic form
Cm = z.reshape(Xm.shape)

Lata, Vata = np.linalg.eig(AtA)

print(Vata)

AtA[:,0] /= np.linalg.norm(AtA[:,0])
AtA[:,1] /= np.linalg.norm(AtA[:,1])

Vata[:,0] /= np.linalg.norm(Vata[:,0])
Vata[:,1] /= np.linalg.norm(Vata[:,1])

plt.scatter(x,y,c=c,s=20,edgecolors='none',cmap='jet')
plt.colorbar()
plt.contour(Xm,Ym,Cm,levels=20,cmap='jet')
plt.arrow(0,0,AtA[0,0],AtA[1,0], color='black', linewidth=1.25, head_width=0.1, head_length=0.05)
plt.arrow(0,0,AtA[0,1],AtA[1,1], color='black', linewidth=1.25, head_width=0.1, head_length=0.05)
plt.arrow(0,0,Vata[0,0],Vata[1,0], color='red', linewidth=1.25, head_width=0.1, head_length=0.05)
plt.arrow(0,0,Vata[0,1],Vata[1,1], color='red', linewidth=1.25, head_width=0.1, head_length=0.05)
plt.show()
