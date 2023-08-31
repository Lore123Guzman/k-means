from sklearn.cluster import KMeans
import numpy as np

# Datos de ejemplo (transacciones)
data = np.array([[100],
                 [120],
                 [50],
                 [2000],
                 [90]])

# Crear y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# Obtener etiquetas de cluster para cada transacción
labels = kmeans.labels_

print("Etiquetas de Cluster:", labels)

'''Explicación:

Importamos la clase KMeans de sklearn.cluster para utilizar el algoritmo K-Means.
Creamos un conjunto de datos de ejemplo data que contiene valores numéricos que representan transacciones.
Creamos una instancia del modelo K-Means con 2 clusters (grupos) usando KMeans(n_clusters=2).
Ajustamos el modelo a los datos utilizando kmeans.fit(data).
Obtenemos las etiquetas de cluster para cada transacción utilizando kmeans.labels_, lo que asigna cada punto a uno de los dos clusters.
Este código ilustra cómo K-Means se puede aplicar para agrupar transacciones similares en categorías y, en este contexto, puede ser utilizado para identificar transacciones que son potencialmente anómalas o fraudulentas.'''