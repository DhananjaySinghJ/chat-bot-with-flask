import os
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from flask import Flask, render_template, request

# Parameters
max_length = 20
vocab_size = 10000

# Load and preprocess the data
def load_conversations():
    lines = open('cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conversations = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(','))

    questions = []
    answers = []

    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i + 1]])

    return questions, answers

questions, answers = load_conversations()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    return tokens

questions = [' '.join(preprocess_text(question)) for question in questions]
answers = [' '.join(preprocess_text(answer)) for answer in answers]

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(questions + answers)

questions_seq = tokenizer.texts_to_sequences(questions)
answers_seq = tokenizer.texts_to_sequences(answers)

max_length = max([len(seq) for seq in questions_seq + answers_seq])
questions_padded = pad_sequences(questions_seq, maxlen=max_length, padding='post')
answers_padded = pad_sequences(answers_seq, maxlen=max_length, padding='post')

X = questions_padded
y = answers_padded

# Create the model
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_length))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=64)

def generate_response(model, tokenizer, input_text, max_length):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=max_length, padding='post')
    predicted_seq = model.predict(input_padded, verbose=0)
    predicted_word_index = np.argmax(predicted_seq)
    predicted_word = tokenizer.index_word.get(predicted_word_index, '')

    response = predicted_word
    for _ in range(10):
        input_seq = tokenizer.texts_to_sequences([response])
        input_padded = pad_sequences(input_seq, maxlen=max_length, padding='post')
        predicted_seq = model.predict(input_padded, verbose=0)
        predicted_word_index = np.argmax(predicted_seq)
        predicted_word = tokenizer.index_word.get(predicted_word_index, '')
        response += ' ' + predicted_word

    return response

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chatbot_index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    response = generate_response(model, tokenizer, user_message, max_length)
    return response

if __name__ == '__main__':
    app.run(debug=True)
