import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

# altere a matriz A para ver as diferentes transformações Ax
A = np.array( [ [1,-1],
                [1,-1] ], dtype='float64' )

n = 20
all_x = np.array( [ np.random.rand(n)*2-1, np.random.rand(n)*2-1 ], dtype='float64' )

all_v = A @ all_x

all_v = all_v.T
all_x = all_x.T

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)  # Make room for the slider

# Initial data
ax.arrow(0,0,A[0,0],A[1,0], color='red', linewidth=1.25, head_width=0.1, head_length=0.05)
ax.arrow(0,0,A[0,1],A[1,1], color='green', linewidth=1.25, head_width=0.1, head_length=0.05)
plt.legend(['1a coluna de A', '2a coluna de A'])
my_plots =  [ ax.arrow(0,0,x[0],x[1], color='blue', head_width=0.1, head_length=0.05) for x in all_x ]

# Set axis limits
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Create slider axis
slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]

# Create slider
my_slider = Slider(
    ax=slider_ax,
    label='factor',
    valmin=0.0,
    valmax=1.0,
    valinit=0.0,
    valstep=0.02
)

# Callback function for slider
def update(val):
    a = my_slider.val
    for v, x, my_plot in zip(all_v,all_x,my_plots):
      this_v = a*v + (1-a)*x
      my_plot.set_data(x=0,y=0,dx=this_v[0],dy=this_v[1], head_width=0.1, head_length=0.05)
    plt.draw()
    plt.grid(True)
    fig.canvas.flush_events()

# Register the callback
my_slider.on_changed(update)

plt.grid(True)
plt.show()
