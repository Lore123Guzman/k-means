import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# Cargar la imagen
image = Image.open("tienda.png")
original_shape = np.array(image).shape

# Convertir la imagen a una matriz de píxeles 2D
image_2d = np.array(image).reshape(-1, 2)

# Aplicar K-Means para comprimir la imagen a 16 colores
n_colors = 16
kmeans = KMeans(n_clusters=n_colors, random_state=0)
kmeans.fit(image_2d)

# Asignar a cada píxel el valor del centroide más cercano
compressed_image_1d = kmeans.cluster_centers_[kmeans.labels_]
compressed_image = compressed_image_1d.reshape(original_shape)

# Mostrar la imagen original y la imagen comprimida
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Imagen Original")
plt.axis("on")

plt.subplot(1, 2, 2)
plt.imshow(compressed_image.astype(np.uint8))
plt.title("Imagen Comprimida ({} colores)".format(n_colors))
plt.axis("off")

plt.show()
