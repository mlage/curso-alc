import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

img = Image.open("smiley.png")
smiley = np.array(img)/255 # normalize to [0,1]

img = Image.open("frowny.png")
frowny = np.array(img)/255 # normalize to [0,1]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)  # Make room for the slider

# Initial data
my_plot = ax.imshow(frowny)

# Set axis limits
ax.set_xlim(0, frowny.shape[1])
ax.set_ylim(frowny.shape[0],0)

# Create slider axis
slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]

# Create slider
my_slider = Slider(
    ax=slider_ax,
    label='smiley factor',
    valmin=0.0,
    valmax=1.0,
    valinit=0.0,
    valstep=0.02
)

# Callback function for slider
def update(val):
    a = my_slider.val
    my_plot.set_data(a*smiley + (1-a)*frowny) # a restricted type of linear combination: a*x + (1-a)*y
    plt.draw()
    fig.canvas.flush_events()

# Register the callback
my_slider.on_changed(update)

plt.show()
