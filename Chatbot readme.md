## Chatbot with Movie Dialogs Dataset

This repository implements a chatbot model using a sequence-to-sequence (seq2seq) architecture with LSTMs. It utilizes the Cornell Movie-Dialogs corpus to train the model for generating responses to user queries. Additionally, a Flask app is included for interacting with the chatbot in a web interface.

**Features:**

* Trains a seq2seq model with LSTMs for chatbot conversation.
* Preprocesses movie dialog data for model training.
* Provides a user-friendly Flask app for chatbot interaction.

**Requirements:**

* Python 3.x
* TensorFlow
* Keras
* NLTK
* Flask

**Running the Chatbot:**

1. Clone this repository.
2. Run the Flask app using `python chatbot.py`.
3. Access the app in your web browser at `http://127.0.0.1:5000/` (default port).
4. Type your message in the chat window and press Enter to receive a response from the chatbot.

**Model Architecture:**

The model consists of:

* **Embedding Layer:** Maps words to numerical vectors.
* **LSTM Layers:** Capture sequential relationships in the conversation.
* **Dense Layer:** Generates the output word probabilities.

**Flask App:**

The Flask app allows users to interact with the chatbot through a web interface. Users can type their message and see the chatbot's response displayed below.

**Further Development:**

* Experiment with different hyperparameters (e.g., number of layers, units) for model improvement.
* Implement beam search for generating more diverse and informative responses.
* Integrate sentiment analysis to provide more context-aware responses.
* Deploy the chatbot to a production environment for wider accessibility.

**Feel free to contribute to this project by:**

* Reporting issues
* Suggesting improvements
* Implementing the ideas mentioned in Further Development

I hope this project provides a valuable example of building a chatbot using a seq2seq model with LSTMs.

**Code:**

The complete code for the chatbot model and Flask app can be found in the following files:

* `Chatbot.py` (main script)

**Note:** Make sure to replace `app.py` and `utils.py` with the actual filenames of your scripts if they differ.
