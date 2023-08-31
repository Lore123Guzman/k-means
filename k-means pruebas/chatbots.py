import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

user_questions = [
    "¿Cómo restablezco mi contraseña?",
    "¿Cuál es el horario de atención al cliente?",
    "Quiero devolver un producto, ¿cómo lo hago?",
    "¿Tienen envío gratuito?",
    "Necesito ayuda con mi factura",
    "¿Cuánto tiempo lleva la entrega?"
]


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(user_questions)
from sklearn.cluster import KMeans

num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)
labels = kmeans.labels_
categories = {}

for i, label in enumerate(labels):
    if label not in categories:
        categories[label] = []
    categories[label].append(user_questions[i])

for category, questions in categories.items():
    print(f"Categoría {category+1}:\n")
    for question in questions:
        print(f"- {question}")
    print("\n")
