from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar archivos necesarios
try:
    words = pickle.load(open("data/words.pkl", "rb"))
    classes = pickle.load(open("data/classes.pkl", "rb"))
    with open("data/intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)
except FileNotFoundError as e:
    raise RuntimeError(f"Archivo no encontrado: {e}")

# Descargar recursos NLTK
nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

# Cargar modelos existentes
models = {}
try:
    # Cargar el modelo SVM y el vectorizador como "Model 1"
    svm_model = pickle.load(open("models/svm_model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    models["Model 1"] = {"model": svm_model, "vectorizer": vectorizer}

    # Cargar el modelo basado en redes neuronales como "Model 2"
    models["Model 2"] = load_model("models/chatbot_model2.h5")
except Exception as e:
    raise RuntimeError(f"Error al cargar los modelos: {e}")

# Cargar el modelo T5 (Model 3)
try:
    tokenizer_t5 = T5Tokenizer.from_pretrained("models/t5-finetuned")
    model_t5 = T5ForConditionalGeneration.from_pretrained("models/t5-finetuned")
    models["Model 3"] = {"tokenizer": tokenizer_t5, "model": model_t5}
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo T5: {e}")

# Modelo actual por defecto
current_model = "Model 1"

# Función para preprocesar oraciones
def clean_up_sentence(sentence):
    """Preprocesar la entrada del usuario."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convertir frases a bolsa de palabras
def bag_of_words(sentence, words):
    """Convertir la frase del usuario a una bolsa de palabras."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Función para predecir clases
def predict_class(sentence, model_data, is_t5=False, is_svm=False):
    """Predecir la clase de la consulta."""
    if is_t5:
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]
        inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True)
        outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, 1.0  # Confianza arbitraria para T5
    elif is_svm:
        vectorizer = model_data["vectorizer"]
        model = model_data["model"]
        vectorized_sentence = vectorizer.transform([sentence])
        prediction = model.predict(vectorized_sentence)[0]
        return prediction, 1.0  # Confianza arbitraria para SVM
    else:
        bow = bag_of_words(sentence, words)
        res = model_data.predict(np.array([bow]))[0]
        max_index = np.argmax(res)
        category = classes[max_index]
        confidence = res[max_index]
        return category, float(confidence)

# Obtener respuesta basada en tag
def get_response(tag):
    """Obtener una respuesta basada en el tag predicho."""
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "Lo siento, no entendí tu consulta. ¿Podrías reformularla?"

# Ruta principal para renderizar la página HTML
@app.route('/')
def home():
    """Cargar la página principal."""
    return render_template('index.html')

# Ruta para obtener la lista de modelos
@app.route('/models', methods=['GET'])
def get_models():
    """Devolver la lista de modelos disponibles."""
    return jsonify(list(models.keys()))

# Ruta para predecir la respuesta
@app.route('/predict', methods=['POST'])
def predict():
    """Obtener predicciones del modelo seleccionado."""
    global current_model
    data = request.json
    message = data.get("message", "")
    selected_model = data.get("model", "Model 1")

    # Cambiar el modelo si es necesario
    if selected_model != current_model:
        current_model = selected_model

    if current_model == "Model 1":
        model_data = models[current_model]
        tag, _ = predict_class(message, model_data, is_svm=True)
        response_text = get_response(tag)
    elif current_model == "Model 3":
        model_data = models[current_model]
        response_text, _ = predict_class(message, model_data, is_t5=True)
    else:
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

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
