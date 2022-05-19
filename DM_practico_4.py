#%% Importación de Libs
import imageio
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2,ifft2,fftshift

#%matplotlib inline

#%% definición de funciones auxiliares
MAT_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                        [0.596,-0.275,-0.321],
                        [0.211,-0.523, 0.311]])

def apply_matrix(img, M):
    return np.matmul(img.reshape((-1,3)), M.T).reshape(img.shape)

def rgb2yiq(img):
    return apply_matrix(img, MAT_RGB2YIQ)

def yiq2rgb(img):
    return apply_matrix(img, np.linalg.inv(MAT_RGB2YIQ))

def plot_hist(im, bins, ax, cumulative=False):
    counts, borders = np.histogram(im if im.ndim==2 else rgb2yiq(im)[...,0], bins=bins, range=(0,1))
    ax.bar(range(len(counts)), np.cumsum(counts) if cumulative else counts)
    plt.xticks(ax.get_xticks(), labels=np.round(ax.get_xticks()/bins,2))
    plt.grid(alpha=0.3)

def rmse(img1, img2):
    return np.sqrt(np.mean((img1-img2)**2))

def my_plot_hist(imgin,bins,ax):
    # Se procesa un array2D, o img gray o img L
    img = imgin if imgin.ndim==2 else rgb2yiq(imgin)[:,:,0]
    vals_hist, x_hist = np.histogram(img,bins);
    vals_hist = vals_hist/np.sum(vals_hist);
    ax.bar(x_hist[0:len(x_hist)-1],vals_hist,width=1/(2*bins))
    ax.set_xlim([-.2,1.2])
    ax.grid(alpha=0.5)
    return x_hist[0:len(x_hist)-1],vals_hist
    # argumentos de entrada: imgin, metodo[minmax,percentil], value
    if len(args) <2: print('Cantidad insuficente de argumentos')
    if len(args)==2: imgin=args[0]; m=args[1];
    if len(args)==3: imgin=args[0];m=args[1];val=args[2] 
    if len(args)>3: print('Error, solamente se pueden ingresar hasta 2 argumentos...')

    if imgin.ndim==2:
        img = imgin;
        print('Se esta procesando una img gray.');
    if imgin.ndim==3:
        img = rgb2yiq(imgin)[:,:,0]; # siempre se procesa un array2D
        print('Se esta procesando una img rgb.');
        print('Se percibe el incremento de luminosidad y la mejora en los detalles.')
        # Se observan mas detalles en la img, aunque alguna componente quedó clampleada :(

    
    # Normalización por max y min del histograma
    if m == 'minmax':     img_norm = norm_hist_minmax(img);
    # Normalización por percentiles
    if m == 'percentil':  img_norm = norm_hist_percentil(img,val);
    # Gamma
    if m == 'gamma':  img_norm = correccion_gamma(img,val);
    # Interpolación
    if m == 'interpolacion':  img_norm = interpolacion(img,val);

    # Para el caso img_3D (probablmente rgb)
    if imgin.ndim==3: # img RGB
        # Img en YIQ con histograma normalizado
        img_norm = L2rgb(img_norm,imgin);

    print('Img final con hist normalizado, max y min: ',
          round(img_norm.max(),2),round(img_norm.min(),2));
    return img_norm;

# %% Primer inciso: Etapa 0 - Para chequear
imgin     = imageio.imread('imageio:chelsea.png')/255
debug = 1;
bins = 100;
name_mod2save = 'mod_save.png';
# Se procesa un array2D, o img gray o img L
img = imgin if imgin.ndim==2 else rgb2yiq(imgin)[:,:,0];

filas,cols = img.shape;
K = 1;#(1/(filas*cols));
img_fft = K*fft2(img);
img_restore = np.real(ifft2(img_fft))/K;
error_rmse = np.abs( rmse(img,img_restore));
print('Test point 0: Chequeo img -> fft ->ifft -------------------------------')
print ('-----------------------------------------------------------------------')
print('error rmse mínimo a conseguir: ',format(error_rmse,'.2E'),'\n')

if debug:
    _, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].imshow(img,'gray',vmin=0, vmax=1); ax[0].set_title('img_fftmg original')
    ax[1].imshow(img_restore,'gray', vmin=0, vmax=1);  ax[1].set_title('img_fftmg después de la ifft2')
    print('Img original: max(), sum(): ', img.max(),np.sum(img));
    print('Img img_restore: max(), sum(): ', img_restore.max(),np.sum(img_restore));
# %% Primer inciso: Etapa 1 - Visualización del espectro en módulo y fase 
img_mod     = np.abs(img_fft);     
img_fase   = np.angle(img_fft);

_, ax1 = plt.subplots(1, 2, figsize=(15,5))
ax1[0].imshow(np.log(fftshift(img_mod)), 'gray'); ax1[0].set_title('Módulo $fft2(Img)$')
ax1[1].imshow(fftshift(img_fase), 'gray');       ax1[1].set_title('Fase $fft2(img)$')
if debug:
    print('max y min en el modulo(sin log()): ',  format(img_mod.max(),'.2E'),' y ',format(img_mod.min(),'.2E'))
    print('max y min en el modulo(con log()): ',  format(np.log(img_mod.max()),'.2E'),' y ',format(np.log(img_mod.min()),'.2E'))
    print('max y min en fase: ', format(img_fase.max(),'.2E'),' y ',format(img_fase.min(),'.2E'))


#%% Segundo y Cuarto inciso: Etapa 0 - Análisis de la distribucion de los valores de magnitud
alfa = 1;
gama=1/20;
img_mod_w = img_mod;#np.abs(img_fft);
img_mod_w_log = np.log(np.abs(img_fft));

yp1,xp1 = np.histogram(img_mod_w,bins);
img_mod_w_gamma = alfa*img_mod_w**gama;
yp2,xp2 = np.histogram(img_mod_w_gamma,bins);
yp3,xp3 = np.histogram(img_mod_w_log,bins);
yp4,xp4 = np.histogram(-img_mod_w_log,bins);

# Graficas
_, ax = plt.subplots(2, 2, figsize=(15,5));
ax[0,0].plot(xp1[0:len(xp1)-1],yp1);    #ax[0,0].set_title('Histograma $|fft2(Img)|$')
ax[0,0].text(.3,.9*ax[0,0].get_ylim()[1], 'Histograma $|fft2(img)|$');

ax[1,0].plot(xp2[0:len(xp1)-1],yp2);    #ax[1,0].set_title('Histograma $|fft2(img_fftmg)|^\gamma$')  
ax[1,0].text(1.2,.9*ax[1,0].get_ylim()[1], 'Histograma $|fft2(img)|^\gamma$');

ax[0,1].plot(xp3[0:len(xp1)-1],yp3);    #ax[0,1].set_title('Histograma $log|fft2(Img)|$')
ax[0,1].text(4,.9*ax[0,1].get_ylim()[1], 'Histograma $log(|fft2(img)|)$');

ax[1,1].plot(xp4[0:len(xp1)-1],yp4);    #ax[1,1].set_title('Histograma $-Log|fft2(Img)|$')
ax[1,1].text(-10,.9*ax[1,1].get_ylim()[1], 'Histograma $-log(|fft2(img)|)$');

# Mini conclusiones
# Visualmente, tanto la corrección gamma, como -log, y log consiguen
# distribuciones similares, prometedoras y sencillas de implementar. 

# Se tiene la idea que una distribución uniforme en los *valores del modulo*,
# conseguiria una *minimo mse*. Dado que las distribuciones mostradas pueden modelarle
# con una gausiana, quedo pendiente usar la cdf de la normal como 2da función
# de transformación.

#%% Segundo y Cuarto inciso: Etapa 1 - Análisis de la distribucion de la fase
img_fase_w = img_fase;
yp1,xp1 = np.histogram(img_fase_w,bins);
yp1 = yp1/np.sum(yp1);
_, ax = plt.subplots(1, 2, figsize=(15,5));
ax[0].plot(xp1[0:len(xp1)-1],yp1); ax[0].set_ylim([0,.1])
ax[0].set_title('pdf fase de $fft(img)$');
ax[1].plot(xp1[0:len(xp1)-1],np.cumsum(yp1)); ax[1].set_title('cdf fase de $fft(img)$');

# Mini conclusiones
# Visualmente, podria decirse que la evidencia indica que la fase de esta imagen
# tiene una ditribución uniforme. Por esta razón no se realiza ninguna transformación
# sobre la fase.


#%% Segundo y Cuarto inciso: Etapa 2 - Transformación y guardado del espectro
#filas,cols = img.shape; K = 1;#(1/(filas*cols)); img_fft = K*fft2(img);

img_mod_w     = np.abs(fft2(img));     

# img_fft transformada con -log()
mod_transf_w = -np.log(img_mod_w);
# img con histograma normalizado
mx = mod_transf_w.max(); mn = mod_transf_w.min();
mod_transf_w = (mod_transf_w-mn)/(mx-mn) 

# Guardado and read
imageio.imwrite(name_mod2save,255*mod_transf_w);
mod_transf_r = imageio.imread(name_mod2save)/255;

print('Test point 1: antes de guardar y despues de leer T{|fft2(img)|} -------')
print ('-----------------------------------------------------------------------')

if debug:
    print('max y min en -log|fft2(I) norm|: ', format(mod_transf_w.max(),'.2E'),' y ',format(mod_transf_w.min(),'.2E'))

    #Graficas
    _, ax = plt.subplots(1, 2, figsize=(15,5));
    yp1,xp1 = np.histogram(img_mod_w,bins);           ax[0].plot(xp1[0:len(xp1)-1],yp1);
    ax[0].set_title( 'Histograma $|fft2(img)|$');   
    yp2,xp2 = np.histogram(mod_transf_w,bins);        ax[1].plot(xp2[0:len(xp2)-1],yp2);
    ax[1].set_title( 'Histograma $-log(|fft2(img)|)$');
    
    _, ax1 = plt.subplots(1, 3, figsize=(15,5));
    ax1[0].imshow(np.log(fftshift(img_mod_w)), 'gray');     ax1[0].set_title('Espectro $Log(|fft2(Img)|)$')
    ax1[1].imshow(fftshift(mod_transf_w), 'gray',vmin=0, vmax=1);  ax1[1].set_title('mod_transf_w');#ax1[1].set_title('Espectro $-log(|fft2(Img)|)$ normalizado')
    ax1[2].imshow(fftshift(mod_transf_r), 'gray',vmin=0, vmax=1);  ax1[2].set_title('mod_transf_r'); #  ax1[2].set_title('Espectro read despues del write')

error_rmse = np.abs( rmse(mod_transf_w,mod_transf_r));
print('error rmse entre mod_transf_w(antes de guardar) y mod_transf_r(despues de leer): ',format(error_rmse,'.2E'),'\n');

#%% Tercer y Cuarto inciso: Etapa 0 - Transformación inversa
# Medicion del modulo del espectro despues de la transformación ^(-1)
# transformación ^(-1)
x = (mx-mn)*mod_transf_r + mn;
img_mod_r = np.exp(-x);
error_rmse = np.abs( rmse(img_mod_w,img_mod_r));

print('Test point 2: antes de T y despues T(-1) ------------------------------')
print ('-----------------------------------------------------------------------')
print('error rmse entre img_mod_w y img_mod_r: ',format(error_rmse,'.2E'),'\n');

if debug:
# Test point
    print('max y min en el img_mod_w(sin log()): ',  format(img_mod_w.max(),'.2E'),' y ',format(img_mod_w.min(),'.2E'));
    print('max y min en el img_mod_r(sin log()): ',  format(img_mod_r.max(),'.2E'),' y ',format(img_mod_r.min(),'.2E'))

    _, ax1 = plt.subplots(1, 2, figsize=(15,5));
    ax1[0].imshow(np.log(fftshift(img_mod_w)), 'gray');     ax1[0].set_title('Espectro img_mod')
    ax1[1].imshow(np.log(fftshift(img_mod_r)), 'gray');     ax1[1].set_title('Espectro img_mod_r');

#%% Tercer y Cuarto inciso: Etapa 1 - modulo y fase ->fft_restore
# math.exp() # es mas rapido
# np.exp() # es mas rapido que math,
#fase_sint = np.pi*uniform.rvs(2.32e-5,1,[filas, cols]);
#img_fft_r = np.multiply( img_mod_w,np.exp(1j*fase_sint) );
img_fft_r = np.multiply( img_mod_w,np.exp(1j*img_fase) );

print('Test point 3: modulo y fase -> fft_r  vs fft(img) inicial -------------')
print ('-----------------------------------------------------------------------')
error_rmse = np.abs( rmse(img_fft,img_fft_r));
print('error rmse entre img_fft y img_fft_r: ',format(error_rmse,'.2E'),'\n');

#%% Tercer y Cuarto inciso: Etapa 2 - nivel pixeles
img_r = ifft2(img_fft_r);
error_rmse = np.abs( rmse(img,img_r));

print('Test point 4: nivel pixeles -------------------------------------------')
print ('-----------------------------------------------------------------------')
print('error rmse a nivel pixel: ',error_rmse);

# Graficas
_, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].imshow(img,'gray');    ax[0].set_title('img original')
ax[1].imshow(np.real(img_r),'gray');  ax[1].set_title('img después de leer y revertir el procedimiento')

#%%
#%% Tercer y Cuarto inciso: Etapa 2 - nivel pixeles
