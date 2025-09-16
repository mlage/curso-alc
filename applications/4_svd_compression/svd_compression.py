import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = np.array(Image.open("flickr_cat_000027.jpg")).astype('float32') / 255

plt.ion()

plt.figure(1)
plt.imshow(img,cmap='Greys_r')
plt.colorbar()
_ = input('continue?')

U, S, Vh = np.linalg.svd(img)

plt.figure(2)
plt.plot(np.linspace(0,S.size-1,S.size),S,'.--')
plt.xlabel("Singular value index")
plt.ylabel("Singular values")
plt.yscale('log')
plt.grid(True)
#_ = input()

cumulative_S = np.cumsum(S)/np.sum(S)

plt.figure(3)
plt.plot(np.linspace(0,S.size-1,S.size),cumulative_S,'.--')
plt.xlabel("Singular value index")
plt.ylabel("Singular values (cumulative sum)")
plt.grid(True)
_ = input('continue?')

Ak = np.zeros_like(img)
errors = []
for i in range(S.size):
    Ak += S[i]* np.outer(U[:,i], Vh[i,:])
    errors.append( np.linalg.norm(img - Ak, 'fro') )
errors = np.array(errors) / np.linalg.norm(img, 'fro')

plt.figure(4)
plt.plot(np.linspace(0,S.size-1,S.size),errors,'.--')
plt.xlabel("Singular value index")
plt.ylabel("||img-Ak||")
plt.yscale('log')
plt.grid(True)
_ = input('continue?')

mask = errors>0.15
img_approx = U[:,mask] @ np.diag(S[mask]) @ Vh[mask,:]
print("15% error, components: ",np.sum(mask))

plt.figure(5)
plt.imshow(img_approx,cmap='Greys_r')
plt.colorbar()
_ = input('continue?')

mask = errors>0.1
img_approx = U[:,mask] @ np.diag(S[mask]) @ Vh[mask,:]
print("10% error, components: ",np.sum(mask))

plt.figure(6)
plt.imshow(img_approx,cmap='Greys_r')
plt.colorbar()
_ = input('continue?')

mask = errors>0.05
img_approx = U[:,mask] @ np.diag(S[mask]) @ Vh[mask,:]
print("5% error, components: ",np.sum(mask))

plt.figure(7)
plt.imshow(img_approx,cmap='Greys_r')
plt.colorbar()
_ = input('end')

plt.ioff()
