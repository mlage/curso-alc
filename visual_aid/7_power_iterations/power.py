import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

#A = np.random.rand(2,2)*2.-1.
#A = A.T @ A
A = np.array([[2,1],[1,2]],dtype='float64')

x = np.random.rand(2)*2.-1.

steps = 40

q = x / np.linalg.norm(x)
q0 = q.copy()
power_iterations = [ q ]
for _ in range(steps):
    q = A @ q
    q /= np.linalg.norm(q)
    power_iterations.append(q)
    
# ground truth
L, V = np.linalg.eigh(A)
max_eigvec = V[:,np.argmax(L)]
max_eigvec /= np.linalg.norm(max_eigvec)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)  # Make room for the slider

# Initial data
ax.arrow(0,0,A[0,0],A[1,0], color='red', linewidth=1.25, head_width=0.1, head_length=0.05)
ax.arrow(0,0,A[0,1],A[1,1], color='green', linewidth=1.25, head_width=0.1, head_length=0.05)
ax.arrow(0,0,max_eigvec[0],max_eigvec[1], color='blue', linewidth=1.25, head_width=0.1, head_length=0.05)
my_plot = ax.arrow(0,0,q0[0],q0[1], color='black', head_width=0.1, head_length=0.05)

# Set axis limits
sz = max( np.max(np.abs(A[0])), np.max(np.abs(A[1])), np.max(np.abs(q0)), np.max(np.abs(max_eigvec)) )
ax.set_xlim(-sz-0.1, sz+0.1)
ax.set_ylim(-sz-0.1, sz+0.1)

# Create slider axis
slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]

# Create slider
my_slider = Slider(
    ax=slider_ax,
    label='iter',
    valmin=0,
    valmax=steps,
    valinit=0,
    valstep=1
)

# Callback function for slider
def update(val):
    a = int(my_slider.val)
    my_plot.set_data(x=0,y=0,dx=power_iterations[a][0],dy=power_iterations[a][1], head_width=0.1, head_length=0.05)
    plt.draw()
    plt.grid(True)
    fig.canvas.flush_events()

# Register the callback
my_slider.on_changed(update)

plt.show()
