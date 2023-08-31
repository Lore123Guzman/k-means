from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Leer las preguntas desde el archivo de texto
with open('lorena.txt', 'r') as file:
    questions = file.readlines()

# Limpiar las preguntas y eliminar duplicados
questions = [question.strip() for question in questions]
unique_questions = list(set(questions))

# Vectorización de las preguntas
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(unique_questions)

# Calcular la matriz de similitud coseno entre las preguntas
similarity_matrix = cosine_similarity(X)

# Definir un umbral para considerar preguntas repetidas
threshold = 0.9

# Encontrar preguntas repetidas
repeated_questions = defaultdict(list)
for i in range(len(unique_questions)):
    for j in range(i + 1, len(unique_questions)):
        if similarity_matrix[i, j] > threshold:
            repeated_questions[unique_questions[i]].append(unique_questions[j])

# Mostrar las preguntas repetidas
for question, repeated_list in repeated_questions.items():
    print("Pregunta:", question)
    print("Preguntas repetidas:")
    for repeated_question in repeated_list:
        print("-", repeated_question)
    print("=" * 50)
