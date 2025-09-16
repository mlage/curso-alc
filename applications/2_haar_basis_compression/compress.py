import numpy as np
from matplotlib import pyplot as plt
import ctypes
import os

# working directory
cwd = os.getcwd()

# ATENÇÃO: Precisa compilar a biblioteca antes de rodar este script!
# gcc -shared -fPIC -o img_compression.so img_compression.c -fopenmp -O3

# variable to reference img_compression.so library
C_LIB = ctypes.CDLL(f'{cwd}/img_compression.so')

# reading original data (float32 image)
rows = 1000
cols = 1000
img = np.fromfile(f'potential_{rows}x{cols}.bin',dtype='float32')

# call compress function from C code
threshold = 0.05 * np.std(img) 
file_compressed = f'potential_{rows}x{cols}_compressed.bin'
filestr = ctypes.c_char_p(file_compressed.encode('utf-8'))
C_LIB.compress( filestr,
                ctypes.c_void_p(img.ctypes.data),
                ctypes.c_uint32(rows),
                ctypes.c_uint32(cols),
                ctypes.c_float(threshold) )
                
# decompress from file
new_img = np.zeros_like(img)
C_LIB.decompress( ctypes.c_void_p(new_img.ctypes.data), filestr )

plt.figure(1)
plt.imshow(img.reshape(rows,cols),cmap='jet')
plt.colorbar()

plt.figure(2)
plt.imshow(new_img.reshape(rows,cols),cmap='jet')
plt.colorbar()

plt.figure(3)
plt.imshow(np.abs(img-new_img).reshape(rows,cols),cmap='Reds')
plt.colorbar()

plt.show()
