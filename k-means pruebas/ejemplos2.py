import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt

# Cargar la imagen
image = Image.open("FOTO.jpg")
image_array = np.array(image)

# Convertir la matriz de píxeles a una matriz 2D
pixels = image_array.reshape(-1, 3)

# Número de colores/clústeres deseados
n_colors = 16

# Crear y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=n_colors, random_state=0)
kmeans.fit(pixels)

# Asignar cada píxel a un clúster
labels = kmeans.predict(pixels)

# Reemplazar cada píxel por el valor del centroide del clúster
compressed_pixels = kmeans.cluster_centers_[labels]

# Reconvertir la matriz de píxeles comprimida a la forma original
compressed_image_array = compressed_pixels.reshape(image_array.shape)

# Crear y mostrar la imagen comprimida
compressed_image = Image.fromarray(compressed_image_array.astype("uint8"))
compressed_image.show()

# Guardar la imagen comprimida
compressed_image.save("compressed_image.jpg")

# Mostrar la imagen original y comprimida
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Compressed Image")
plt.imshow(compressed_image)
plt.axis("off")

plt.show()
