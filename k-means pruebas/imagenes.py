import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D

# Función para cargar y preparar la imagen
def load_and_prepare_image(image_path):
    image = imageio.imread(image_path)
    image_reshaped = image.reshape(-1, 3)
    return image, image_reshaped

# Cargar y preparar la imagen desde diferentes ángulos
image_front, image_reshaped_front = load_and_prepare_image("FOTO.jpg")
image_side, image_reshaped_side = load_and_prepare_image("tienda.png")

# Número de colores/clústeres deseados
n_colors = 8

# Crear y entrenar el modelo K-Means para la vista frontal
kmeans_front = KMeans(n_clusters=n_colors, random_state=0)
kmeans_front.fit(image_reshaped_front)
labels_front = kmeans_front.predict(image_reshaped_front)
compressed_pixels_front = kmeans_front.cluster_centers_[labels_front]

# Crear y entrenar el modelo K-Means para la vista lateral
kmeans_side = KMeans(n_clusters=n_colors, random_state=0)
kmeans_side.fit(image_reshaped_side)
labels_side = kmeans_side.predict(image_reshaped_side)
compressed_pixels_side = kmeans_side.cluster_centers_[labels_side]

# Crear y mostrar gráficos 3D de los clústeres
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(image_reshaped_front[:, 0], image_reshaped_front[:, 1], image_reshaped_front[:, 2], c=labels_front)
ax1.set_title('Vista foto Clusters')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(image_reshaped_side[:, 0], image_reshaped_side[:, 1], image_reshaped_side[:, 2], c=labels_side)
ax2.set_title('Vista Tienda Clusters')

plt.tight_layout()
plt.show()
