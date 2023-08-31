import cv2
import numpy as np

# Cargar la imagen del rostro
image = cv2.imread('foto.jpeg')

# Convertir la imagen a formato RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redimensionar la imagen para facilitar el procesamiento
resized_image = cv2.resize(image, (500, 500))

# Convertir la imagen redimensionada en un arreglo de píxeles
pixels = resized_image.reshape((-1, 3))

# Convertir los valores de los píxeles a tipo float32
pixels = np.float32(pixels)

# Definir los parámetros para el algoritmo K-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 5  # Número de clusters

# Aplicar el algoritmo K-means
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convertir los centros de los clusters a valores enteros
centers = np.uint8(centers)

# Asignar cada píxel al cluster correspondiente
segmented_image = centers[labels.flatten()]

# Restaurar la forma original de la imagen
segmented_image = segmented_image.reshape(resized_image.shape)

# Mostrar la imagen original y la imagen segmentada
cv2.imshow('Original', resized_image)
cv2.imshow('Segmentada', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
