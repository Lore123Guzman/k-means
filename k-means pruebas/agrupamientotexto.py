import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

# Crea un conjunto de datos de ejemplo
data = {
    'texto': [
        'Este es un ejemplo de procesamiento de lenguaje natural.',
        'El aprendizaje automático es emocionante.',
        'El análisis de texto es parte del NLP.',
        'K-means es un algoritmo de agrupamiento.',
        'El procesamiento de lenguaje natural es importante en la IA.'
    ]
}

# Crea un DataFrame a partir del conjunto de datos
df = pd.DataFrame(data)

# Vectorización de TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['texto'])

# Reducción de dimensionalidad con Truncated SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# Aplicación de K-means
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X_svd)

# Agrega las etiquetas de cluster al DataFrame
df['cluster_label'] = kmeans.labels_

# Visualización de los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=X_svd[:, 0], y=X_svd[:, 1], hue='cluster_label', palette='Set2')
plt.title('Resultado de K-means Clustering en Documentos de Texto')
plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.legend()
plt.show()
