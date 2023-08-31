import numpy as np # libreria para hacer matriz
import matplotlib.pyplot as plt #crear arreglos 
from sklearn.cluster import KMeans #k-means
import cv2 as cv #libreria opencv imagenes 


#se carga la imagen 
I = cv.imread("tienda.jpg")


I1 = np.asarray(I,dtype=np.float32)/255
plt.figure(figsize=(12,12))
plt.imshow(I1)
plt.axis('off')
plt.show()

#Es una imagen BGR. Intercambiamos los canale R y B para obtener una imagen RGB

I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

#se convierte a una matriz 
a = np.asarray(I,dtype=np.float32)/255
plt.figure(figsize=(12,12))
plt.imshow(a)
plt.axis('off')
plt.show()

#Primero obtenemos el número de filas, columnas y canales de a.. 
# El primero nos dará la altura en píxeles de la imagen h y el segundo la anchura w.

#El número de píxeles de la imagen es w*h

h, w, c = a.shape
print('w =', w)
print('h =', h)
print('c =', c)
num_pixels = w*h 
print ('Número de pixels = ', num_pixels)

#Creamos un array con tantas filas como píxeles y por cada fila/pixel, 
#Se crean 3 columnas , una para cada intesidad de color (rojo, verde y azul)
a1 = a.reshape(w*h, c)

print('filas, columnas y canales de a ', a.shape)
print('filas y columnas de  a1', a1.shape)

colores = np.unique(a1, axis=0, return_counts=True)
print(colores)

num_colores = colores[0].shape[0]
print ('\nNumeero de colores = ', num_colores)

n = 60
k_means = KMeans(n_clusters=n)
model = k_means.fit(a1)

#cambiar dimensiones, etiquetas y centroides 
centroides = k_means.cluster_centers_
etiquetas = k_means.labels_

print('dimensiones centroides ', centroides.shape)
print('dimensiones etiquetas ', etiquetas.shape)

a2k = centroides[etiquetas]
print('Dimensiones matriz a2k ', a2k.shape)

a3k = a2k.shape(h,w,c)
print('Dimensiones matriz a3k ', a3k.shape)

plt.figure(figsize=(12,12))
plt.imshow(a3k)
plt.axis('off')
plt.show()

a4k = np.floor(a3k*255)
a5k = a4k.astype(np.uint8)

red = np.copy(a5k[:,:,1])
blue = np.copy(a5k[:,:,2])

a5k[:,:,0] = blue
a5k[:,:,2] = red

Ik = cv.imwrite("tienda.jpg",a5k)

coloresk = np.unique(a2k, axis=0, return_counts=True)
num_coloresk = coloresk[0].shape[0]
num_pixelsk = a2k.shape[0]
print('Número de píxeles  = ',num_pixelsk )
print('Número de colores  = ', num_coloresk)