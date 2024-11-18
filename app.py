from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import json
import os

app = Flask(__name__)

if os.path.exists("data/words.pkl") and os.path.exists("data/classes.pkl"):
    words = pickle.load(open("data/words.pkl", "rb"))
    classes = pickle.load(open("data/classes.pkl", "rb"))
else:
    raise FileNotFoundError("Los archivos 'words.pkl' o 'classes.pkl' no se encontraron en la carpeta 'data'.")

if os.path.exists("data/intents.json"):
    with open("data/intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)
else:
    raise FileNotFoundError("El archivo 'intents.json' no se encontró en la carpeta 'data'.")

nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

# Verificar si los modelos existen antes de cargarlos
models = {}
try:
    models["Model 1"] = load_model("models/chatbot_model.h5")
    models["Model 2"] = load_model("models/chatbot_model2.h5")
except Exception as e:
    raise RuntimeError(f"Error al cargar los modelos: {e}")

current_model = "Model 2"  # Modelo por defecto


def clean_up_sentence(sentence):
    """Preprocesar la entrada del usuario."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    """Convertir la frase del usuario a una bolsa de palabras."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    """Predecir la clase de la consulta."""
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    category = classes[max_index]
    confidence = res[max_index]
    return category, float(confidence)


def get_response(tag):
    """Obtener una respuesta basada en el tag predicho."""
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "Lo siento, no entendí tu consulta. ¿Podrías reformularla?"


@app.route('/')
def home():
    """Cargar la página principal."""
    return render_template('index.html')


@app.route('/models', methods=['GET'])
def get_models():
    """Devolver la lista de modelos disponibles."""
    return jsonify(list(models.keys()))


@app.route('/predict', methods=['POST'])
def predict():
    """Obtener predicciones del modelo seleccionado."""
    global current_model
    data = request.json
    message = data.get("message", "")
    selected_model = data.get("model", "Model 2")

    # Cambiar el modelo si es necesario
    if selected_model != current_model:
        current_model = selected_model

    model = models[current_model]
    tag, confidence = predict_class(message, model)

    if confidence > 0.7:
        response_text = get_response(tag)
    else:
        response_text = "Lo siento, no entendí tu consulta. ¿Podrías reformularla?"

    response = {
        "response": response_text,
        "model": current_model
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
