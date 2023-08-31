import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns 
#lobrerias seaborn y matplotlib para crear matriz y segmentaciones 

# Cargar los datos desde un archivo Excel
data = pd.read_excel('datos Periferia.xlsx', sheet_name='Talent_PreCollaborator')

# Seleccionar las columnas relevantes para el clustering (por ejemplo, características numéricas)
selected_columns = ['feature1', 'feature2', 'feature3']
X = data[selected_columns]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X_scaled)

# Agregar las etiquetas de cluster al DataFrame
data['cluster_label'] = kmeans.labels_

# Visualización de los resultados
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='feature1', y='feature2', hue='cluster_label', palette='Set2')
plt.title('Resultado de K-means Clustering en Datos')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
