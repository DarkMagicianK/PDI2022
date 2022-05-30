#%% Quinto inciso: 

import imageio
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2,ifft2,fftshift

img = imageio.imread('tp4.png')/255;
# plt.imshow(img,'gray');
img_fft = fftshift(fft2(img));
img_mod = np.log( np.abs(img_fft) );
img_fase = np.angle(img_fft);

#%%
_, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].imshow(img_mod, 'gray'); ax[0].set_title('Módulo original')
ax[1].imshow(img_fase, 'gray');       ax[1].set_title('Fase original')
mx = img_mod.max(); mn = img_mod.min();
img_mod_norm = 255*(img_mod-mn)/(mx-mn); 

imageio.imwrite('tp4_amp.png',img_mod_norm.astype(np.uint8));
imageio.imwrite('tp4_phase.png',img_fase);
#%%
img_mod_r = imageio.imread('tp4_amp_filter.png')[...,0]/255;
img_mod_filter = (mx-mn)*img_mod_r + mn;
img_fft_r = np.multiply( img_mod_filter,np.exp(1j*img_fase) );
img_restore = np.real(ifft2(fftshift(img_fft_r)));

_, ax1 = plt.subplots(1, 2, figsize=(15,5))
ax1[0].imshow(img_mod_filter, 'gray'); ax1[0].set_title('Módulo corregido')
ax1[1].imshow(img_restore, 'gray');       ax1[1].set_title('Imagen corregida')
