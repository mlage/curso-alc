from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def load_image(filename):
    img = Image.open(filename)
    arr = np.array(img)
    rows, cols, _ = arr.shape
    dreal, dimag = cols/2, rows/2
    markers = []
    for y,row in enumerate(arr):
        markers.extend( [ ( (-dreal+x)  + (dimag - y)*1j, tuple(pixel/255) ) for x,pixel in enumerate(row)] )
    return markers

def plot_image(data):
    p = plt.scatter( [ d.real for d,_ in data ], [ d.imag for d,_ in data ], c = [ c for _,c in data ], edgecolors=None , marker="s" )
    return p

if __name__ == '__main__':
    
    data = load_image("fish.jpg")
    plt.figure(1)
    plot_image(data)
    
    theta = 45*np.pi/180 # 45 graus para radianos
    data_rot = [ ( z*np.exp(1j*theta), color ) for z, color in data  ]
    
    plt.figure(2)
    plot_image(data_rot)
    
    plt.show()
