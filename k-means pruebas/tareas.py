from sklearn.cluster import KMeans
import numpy as np

# Datos de ejemplo (horas dedicadas a tareas de desarrollo por día)
# Cada fila representa un día y las columnas representan diferentes tareas
data = np.array([[2, 1, 3, 0],
                 [0, 0, 4, 1],
                 [1, 2, 2, 0],
                 [3, 1, 1, 1],
                 [0, 0, 5, 0]])

# Crear y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# Obtener etiquetas de cluster para cada día
labels = kmeans.labels_

print("Etiquetas de Cluster:", labels)
