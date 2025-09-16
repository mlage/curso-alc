import numpy as np
from matplotlib import pyplot as plt

n_data = 1000
n_atts = 2 # fixed for 2D vis

A = np.zeros((n_data,n_atts),dtype='float64')
A[:,0] = np.linspace(1,10,n_data)
A[:,1] = 0.5*A[:,0] + 2.0
A[:,:] += np.random.rand(n_data,n_atts)*0.5
A[n_data//4:-n_data//4,:] += np.random.rand(n_data//2,n_atts)-0.5

U0, S0, Vt0 = np.linalg.svd(A)

mean_data = np.mean(A,axis=0)
A_hat = A - mean_data
U, S, Vt = np.linalg.svd(A_hat)

plt.arrow(mean_data[0],mean_data[1],Vt[0,0],Vt[0,1], color='red', linewidth=2.5, head_width=0.5, head_length=0.25)
plt.arrow(mean_data[0],mean_data[1],Vt[1,0],Vt[1,1], color='red', linewidth=2.5, head_width=0.5, head_length=0.25)

plt.plot([mean_data[0]-4*Vt[0,0], mean_data[0]+4*Vt[0,0]], [mean_data[1]-4*Vt[0,1], mean_data[1]+4*Vt[0,1]], 'k--')
plt.plot([mean_data[0]-4*Vt[1,0], mean_data[0]+4*Vt[1,0]], [mean_data[1]-4*Vt[1,1], mean_data[1]+4*Vt[1,1]], 'k--')

plt.plot(A[:,0],A[:,1],marker='.',linestyle='None',alpha=0.5)

plt.axis('equal')
plt.grid(True)
plt.show()
