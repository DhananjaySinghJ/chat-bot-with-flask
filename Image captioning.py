import os
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
from flask import Flask, render_template, request

# Parameters
img_size = (224, 224)
vocab_size = 8000
max_length = 32
embedding_dim = 256
lstm_units = 256

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet', include_top=False)

# Function to preprocess images
def preprocess_image_for_inference(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        in_text += ' ' + word
        if word == 'end':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            img_path = 'uploaded_image.jpg'
            file.save(img_path)
            img = preprocess_image_for_inference(img_path)
            img_features = model.layers[1].predict(img)
            caption = generate_desc(model, tokenizer, img_features, max_length)
            return render_template('image_captioning_index.html', caption=caption)
    return render_template('image_captioning_index.html')

if __name__ == '__main__':
    app.run(debug=True)
