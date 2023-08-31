from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

# Imagen de muestra
china = load_sample_image("tienda.png") / 255.0
china_shape = china.shape

# Redimensionar la imagen a una matriz 2D
china_2d = china.reshape(china_shape[0] * china_shape[1], china_shape[2])

# Aplicar K-Means para reducir la imagen a 16 colores
kmeans = KMeans(n_clusters=20)
kmeans.fit(china_2d)

# Asignar cada píxel a un cluster y reconstruir la imagen
china_recolored = kmeans.cluster_centers_[kmeans.labels_].reshape(china_shape)

# Mostrar la imagen original y la imagen recoloreada
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(china)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(china_recolored)
plt.title("Imagen Recoloreada")
plt.axis("off")

plt.show()
