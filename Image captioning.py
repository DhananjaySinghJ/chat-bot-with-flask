import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, add
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
import cv2
import nltk
from nltk.corpus import stopwords

# For Flask
from flask import Flask, render_template, request, jsonify

# Download NLTK data
nltk.download('stopwords')

# Parameters
img_size = (224, 224)
vocab_size = 8000
max_length = 32
embedding_dim = 256
lstm_units = 256

def load_images(dir_path):
    images = []
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    return np.array(images)

def load_captions(filename):
    with open(filename, 'r') as f:
        doc = f.read()

    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in descriptions:
            descriptions[image_id] = list()
        descriptions[image_id].append(image_desc)
    return descriptions

def clean_descriptions(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(all_desc)
    return tokenizer

def preprocess_captions(descriptions, tokenizer, max_length):
    vocab_size = len(tokenizer.word_index) + 1
    X = tokenizer.texts_to_sequences(all_desc)
    X = pad_sequences(X, maxlen=max_length, padding='post')
    return X, vocab_size

# Load data
train_images = load_images('flickr8k/images')
train_descriptions = load_captions('flickr8k/text/Flickr8k.token.txt')
tokenizer = clean_descriptions(train_descriptions)
X, vocab_size = preprocess_captions(train_descriptions, tokenizer, max_length)

# Create a mapping from image id to index
train_ids = list(train_descriptions.keys())

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet', include_top=False)

# Extract features from images
def extract_features(imgs):
    features = model.predict(imgs, verbose=1)
    return features

# Extract features for training images
train_features = extract_features(train_images)

# Create training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val, train_ids_train, train_ids_val = train_test_split(train_features, X, train_ids, test_size=0.2, random_state=42)

def create_model(vocab_size, max_length):
    # Encoder model
    input_img = Input(shape=(2048,))
    fe = Dense(256, activation='relu')(input_img)
    fe = Dropout(0.2)(fe)
    fe = Dense(256, activation='relu')(fe)

    # Decoder model
    decoder_input = Input(shape=(max_length,))
    dec_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_input)
    dec_lstm1 = LSTM(lstm_units, return_sequences=True)(dec_emb, initial_state=[fe, fe])
    dec_lstm2 = LSTM(lstm_units)(dec_lstm1)
    decoder_outputs = Dense(vocab_size, activation='softmax')(dec_lstm2)
    model = Model(inputs=[input_img, decoder_input], outputs=decoder_outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

# Create the model
model = create_model(vocab_size, max_length)

def data_generator(descriptions, photos, tokenizer, max_length, batch_size):
    while 1:
        for i in range(len(photos) // batch_size):
            start, end = i * batch_size, (i + 1) * batch_size
            X, y = list(), list()
            for j in range(start, end):
                photo = photos[j]
                in_text = descriptions[train_ids[j]]
                for desc in in_text:
                    seq = tokenizer.texts_to_sequences([desc])[0]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        X.append(photo)
                        y.append(out_seq)
            yield np.array(X), np.array(y)

# Train the model
epochs = 10
batch_size = 64
steps_per_epoch = len(X_train) // batch_size

for i in range(epochs):
    generator = data_generator(train_descriptions, X_train, tokenizer, max_length, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)

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

def preprocess_image_for_inference(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

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
            return render_template('index.html', caption=caption)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
