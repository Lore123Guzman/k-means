from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Ejemplo de respuestas generadas por el chatbot
chatbot_responses = [
    "Hola, ¿en qué puedo ayudarte?",
    "Lamento escuchar eso. ¿Cómo puedo resolver tu problema?",
    "¡Gracias por visitarnos!",
    "Estamos aquí para ayudarte. ¿En qué puedo asistirte?",
    "Por supuesto, estaré encantado de ayudarte.",
    "¿Hay algo más en lo que pueda ayudarte?",
    "Espero que hayas tenido una buena experiencia con nosotros.",
    "¡Bienvenido de nuevo! ¿En qué puedo colaborar hoy?"
]

# Crear una matriz TF-IDF para representar las respuestas como características numéricas
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(chatbot_responses)

# Aplicar K-means para agrupar respuestas similares
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Encontrar las respuestas más representativas de cada clúster
cluster_centers = kmeans.cluster_centers_
distances = np.linalg.norm(X - cluster_centers[kmeans.labels_], axis=1)

# Definir umbral para identificar respuestas repetitivas
threshold = np.percentile(distances, 95)

# Identificar respuestas repetitivas basadas en el umbral
repetitive_indices = np.where(distances > threshold)[0]

print("Índices de respuestas repetitivas:")
for idx in repetitive_indices:
    print(f"Respuesta: '{chatbot_responses[idx]}'")
