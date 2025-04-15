import os
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Descargar recursos
#nltk.download('punkt')
#nltk.download('stopwords')

# Configuración
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
#Elimina las etiquetas HTML y convierte el texto a minúsculas
def preprocess(text):
    text = re.sub(r"<.*?>", " ", text)   
    tokens = word_tokenize(text)
    words = [w.lower() for w in tokens if w.isalpha()]
    words = [w for w in words if w not in stop_words]
    stemmed = [stemmer.stem(w) for w in words]
    return ' '.join(stemmed)
#Carga las reseñas desde una carpeta
def load_reviews_from_folder(folder_path, label):
    reviews = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                reviews.append((content, label))
    return reviews
#Procesa una oración y predice su sentimiento
def predecir_sentimiento(oracion):
    texto_procesado = preprocess(oracion)
    vector = vectorizer.transform([texto_procesado])
    prediccion = model.predict(vector)[0]
    return prediccion

base_data = r"C:\Users\SuperUsuario\Desktop\proyectos\aclImdb_v1\aclImdb\test"

neg_rev = load_reviews_from_folder(os.path.join(base_data, 'neg'), 'neg')
pos_rev = load_reviews_from_folder(os.path.join(base_data, 'pos'), 'pos')

documents = neg_rev + pos_rev
random.shuffle(documents)

texts = [preprocess(text) for text, label in documents]
labels = [label for _, label in documents]

# Crear el vectorizador TF-IDF
# Usa las 5000 palabras más relevantes
vectorizer = TfidfVectorizer(max_features=5000)  
X = vectorizer.fit_transform(texts)
# Dividir datos: 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Entrenar modelo LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predecir en el set de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
# Ejemplos
print(predecir_sentimiento("This movie was a total masterpiece! I loved it."))
print(predecir_sentimiento("It was boring, long, and absolutely terrible."))
print(predecir_sentimiento("The acting was great, but the plot was weak."))
print(predecir_sentimiento("I would not recommend this movie to anyone."))