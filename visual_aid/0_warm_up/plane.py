import numpy as np
import matplotlib.pyplot as plt

# column space
c1 = np.array([1,2,0],dtype='float32')
c2 = np.array([2,1,2],dtype='float32')
c3 = c1+c2 #np.array([0,0,0],dtype='float32')
#c3 = np.array([-1,3,1],dtype='float32')

#c1 = np.array([5,2,4],dtype='float32')
#c2 = np.array([3,1,1],dtype='float32')
#c3 = np.array([8,3,5],dtype='float32')

#c1 = np.array([1,0,0],dtype='float32')
#c2 = np.array([0,1,0],dtype='float32')
#c3 = np.array([0,0,1],dtype='float32')

# normalize vectors
#c1 /= np.linalg.norm(c1)
#c2 /= np.linalg.norm(c2)
#c3 /= np.linalg.norm(c3)

# sample points
N = 4000
x = np.random.random((N,3))*2.0 - 1.0
coords = np.zeros((N,3),dtype='float32')
coords[:,0] = x[:,0]*c1[0] + x[:,1]*c2[0] + x[:,2]*c3[0]
coords[:,1] = x[:,0]*c1[1] + x[:,1]*c2[1] + x[:,2]*c3[1]
coords[:,2] = x[:,0]*c1[2] + x[:,1]*c2[2] + x[:,2]*c3[2]

# Create a figure
fig = plt.figure()

# Create a 3D axis object
ax = fig.add_subplot(111, projection='3d')

# Create the scatter plot
ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='r', marker='o')

# Set labels for axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Display the plot
plt.show()
