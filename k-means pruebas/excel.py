import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo Excel
excel_file = 'data (20).xlsx'
sheet_name = 'Sheet1'
data = pd.read_excel(excel_file, sheet_name=sheet_name)

# Preprocesar los datos (esto depende de cómo estén estructurados tus datos)
corpus = data['texto_columna'].tolist()

# Crear una representación vectorial de los datos (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Aplicar el algoritmo K-Means
num_clusters = 5  # Número de clusters deseado
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Agregar las etiquetas de cluster al DataFrame original
data['cluster'] = kmeans.labels_

# Contar cuántos elementos hay en cada cluster
cluster_counts = data['cluster'].value_counts()

# Imprimir los resultados
print("Número de elementos en cada cluster:")
print(cluster_counts)

# Visualización de los resultados (solo para análisis de ejemplo)
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Número de Elementos')
plt.title('Distribución de elementos en cada cluster')
plt.show()
