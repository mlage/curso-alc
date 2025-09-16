import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import QR

#A = np.random.rand(2,2)*2.-1.
#A = A.T @ A
A = np.array([[2,1],[1,2]],dtype='float64')

steps = 40

q = np.eye(A.shape[0])
q0 = q.copy()
q_cumulative = q0.copy()
Ak = A.copy()
power_iterations = [ q ]
my_eigenvalues = [ Ak ]
for _ in range(steps):
    q, r = QR.standard(Ak) # np.linalg.qr(Ak)
    Ak = r @ q
    q_cumulative = q_cumulative @ q
    power_iterations.append(q_cumulative)
    my_eigenvalues.append(Ak)
    
# ground truth
L, V = np.linalg.eigh(A)
V /= np.linalg.norm(V,axis=0)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)  # Make room for the slider

# Initial data
ax.arrow(0,0,A[0,0],A[1,0], color='red', linewidth=1.25, head_width=0.1, head_length=0.05)
ax.arrow(0,0,A[0,1],A[1,1], color='green', linewidth=1.25, head_width=0.1, head_length=0.05)
ax.arrow(0,0,V[0,0],V[1,0], color='blue', linewidth=1.25, head_width=0.1, head_length=0.05)
ax.arrow(0,0,V[0,1],V[1,1], color='blue', linewidth=1.25, head_width=0.1, head_length=0.05)
my_plots = [ ax.arrow(0,0,q0[0,0],q0[1,0], color='black', head_width=0.1, head_length=0.05),
             ax.arrow(0,0,q0[0,1],q0[1,1], color='black', head_width=0.1, head_length=0.05) ]

# Set axis limits
sz = max( np.max(np.abs(A[0])), np.max(np.abs(A[1])), np.max(np.abs(q0)), np.max(np.abs(V)) )
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
    my_plots[0].set_data(x=0,y=0,dx=power_iterations[a][0,0],dy=power_iterations[a][1,0], head_width=0.1, head_length=0.05)
    my_plots[1].set_data(x=0,y=0,dx=power_iterations[a][0,1],dy=power_iterations[a][1,1], head_width=0.1, head_length=0.05)
    plt.draw()
    fig.canvas.flush_events()
    print(f'(step {a}):\nmy_eigenvalues = {my_eigenvalues[a]},\nL = {L}')

# Register the callback
my_slider.on_changed(update)

plt.show()
