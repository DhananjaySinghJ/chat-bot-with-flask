import nltk
from nltk.corpus import movie_dialogs
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from flask import Flask, render_template, request

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

# Load movie dialogs
lines = movie_dialogs.lines()

# Create a DataFrame
conversations = movie_dialogs.conversations()
df = pd.DataFrame({'lineID': [line[0] for line in lines],
                   'characterID': [line[1] for line in lines],
                   'movieID': [line[2] for line in lines],
                   'character': [line[3] for line in lines],
                   'text': [line[4] for line in lines]})

# Merge with conversations to get conversation context
df['conversationID'] = [conv[0] for conv in conversations]
df['utterance_number'] = [conv.index(lineID) for conv, lineID in zip(conversations, df['lineID'])]

# Group by conversation and create a new column with concatenated utterances
df_grouped = df.groupby('conversationID')['text'].apply(lambda x: ' '.join(x)).reset_index()

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df_grouped['text'])

sequences = tokenizer.texts_to_sequences(df_grouped['text'])
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Create input and output sequences
X, y = padded_sequences[:-1], padded_sequences[1:]

# Create the model
model = Sequential()
model.add(Embedding(10000, 64, input_length=max_length))
model.add(LSTM(64))
model.add(Dense(10000, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=64)

def chatbot_response(user_input):
    # Preprocess user input
    user_input_seq = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_seq, maxlen=max_length)

    # Generate response
    predicted_index = np.argmax(model.predict(user_input_padded))
    predicted_word = tokenizer.index_word[predicted_index]

    # Generate more words (iterative approach)
    for _ in range(10):  # Adjust the number of words to generate
        input_seq = np.append(user_input_padded[0][1:], predicted_index)
        input_seq = np.reshape(input_seq, (1, max_length))
        predicted_index = np.argmax(model.predict(input_seq))
        predicted_word += ' ' + tokenizer.index_word[predicted_index]

    return predicted_word

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    chatbot_response = chatbot_response(user_message)
    return chatbot_response

if __name__ == '__main__':
    app.run(debug=True)
