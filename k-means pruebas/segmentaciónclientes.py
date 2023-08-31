from sklearn.cluster import KMeans
import numpy as np

# Datos de ejemplo (historiales de compra)
data = np.array([[10, 15],
                 [12, 20],
                 [25, 30],
                 [5, 10],
                 [30, 35]])

# Crear y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

# Obtener etiquetas de cluster para cada punto
labels = kmeans.labels_

print("Etiquetas de Cluster:", labels)

'''Importamos la clase KMeans de sklearn.cluster.
Creamos un conjunto de datos de ejemplo data que representa historiales de compra de clientes.
Creamos una instancia del modelo K-Means con 2 clusters (grupos) usando KMeans(n_clusters=2).
Ajustamos el modelo a los datos utilizando kmeans.fit(data).
Obtenemos las etiquetas de cluster para cada cliente utilizando kmeans.labels_, lo que agrupa a los clientes según sus patrones de compra.
Este código ilustra cómo K-Means se puede aplicar para segmentar clientes en grupos con patrones de compra similares. Cada grupo puede representar un segmento distinto de clientes, lo que permite a las empresas personalizar sus estrategias de marketing y servicio.'''