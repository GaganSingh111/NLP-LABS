from google.colab import drive
drive.mount('/content/drive')

import keras
import json
from datetime import datetime
import numpy as np
from statistics import median
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.layers import Input, Dense, RepeatVector, LSTM, Embedding, TimeDistributed, Bidirectional
from keras.models import Model

# Path to your dataset
dialogues_path = "/content/drive/MyDrive/Datasets/movie_lines.txt"

VOCAB_SIZE = 5000  # Maximum vocabulary size
EMBEDDING_DIM = 500
EOS_TOKEN = "~e"  # End of sentence token

print("Vocabulary Size:", VOCAB_SIZE)

# Load and preprocess dialogue lines
dialogue_lines = []
with open(dialogues_path, 'r', encoding='utf-8') as dialogues_file:
    for line in dialogues_file:
        line = line.strip().lower()
        split_line = line.split(' +++$+++ ')
        try:
            dialogue_lines.append(split_line[4] + " " + EOS_TOKEN)
        except IndexError:
            # Skip lines that do not have enough fields
            continue

print("Sample dialogues:", dialogue_lines[:5])

# Initialize Keras tokenizer with filters
keras_tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
keras_tokenizer.fit_on_texts(dialogue_lines)

# Convert dialogues to sequences of word indices
text_sequences = keras_tokenizer.texts_to_sequences(dialogue_lines)[:2000]  # Limit to first 2000 samples

# Calculate the median sequence length for padding
MAX_SEQUENCE_LENGTH = int(median(len(seq) for seq in text_sequences))
print("Max sequence length:", MAX_SEQUENCE_LENGTH)

# Pad sequences to fixed length
x_train = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post', value=0)
print("Shape of x_train:", x_train.shape)

# Create one-hot encoded targets for each timestep (needed for categorical_crossentropy)
x_train_rev = []
for x_vector in x_train:
    # For each word index in the sequence, create a one-hot vector of size VOCAB_SIZE
    x_rev_vector = np.zeros((MAX_SEQUENCE_LENGTH, VOCAB_SIZE))
    for i, index in enumerate(x_vector):
        if index > 0 and index < VOCAB_SIZE:  # avoid zero-padding index
            x_rev_vector[i, index-1] = 1  # -1 because indices start at 1
    x_train_rev.append(x_rev_vector)

x_train_rev = np.array(x_train_rev)
print("Shape of x_train_rev (one-hot targets):", x_train_rev.shape)


# Define the sequence-to-sequence model
def get_seq2seq_model():
    main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input')

    embed_1 = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM,
                        mask_zero=True, input_length=MAX_SEQUENCE_LENGTH)(main_input)

    lstm_1 = Bidirectional(LSTM(256, name='lstm_1'))(embed_1)

    repeat_1 = RepeatVector(MAX_SEQUENCE_LENGTH, name='repeat_1')(lstm_1)

    lstm_3 = Bidirectional(LSTM(256, return_sequences=True, name='lstm_3'))(repeat_1)

    softmax_1 = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(lstm_3)

    model = Model(main_input, softmax_1)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Instantiate model
seq2seq_model = get_seq2seq_model()
seq2seq_model.summary()

# Train model
seq2seq_model.fit(x_train, x_train_rev, batch_size=128, epochs=20)

# Prepare index to word mapping for decoding predictions
index2word_map = {v: k for k, v in keras_tokenizer.word_index.items() if v < VOCAB_SIZE}
# Note: tokenizer indices start at 1, and you used index-1 in one-hot, so mapping adjusted accordingly

# Function to decode prediction vectors to word sequence
def sequence_to_str(sequence):
    word_list = []
    for element in sequence:
        index = np.argmax(element) + 1  # Add 1 to revert offset
        if index == 0 or index > VOCAB_SIZE:
            continue
        word = index2word_map.get(index, '')  # fallback to empty string if not found
        if word == EOS_TOKEN:
            break
        word_list.append(word)
    return ' '.join(word_list)


# Generate predictions on training data and save to file
predictions = seq2seq_model.predict(x_train)

predictions_file_path = "/content/" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".txt"

with open(predictions_file_path, 'w', encoding='utf-8') as predictions_file:
    for i in range(len(predictions)):
        predicted_sentence = sequence_to_str(predictions[i])
        actual_sentence = dialogue_lines[i].replace(EOS_TOKEN, '').strip()

        sent_dict = {
            "actual": actual_sentence,
            "generated": predicted_sentence
        }
        predictions_file.write(json.dumps(sent_dict, sort_keys=True, indent=2))
        predictions_file.write("\n")

print(f"Predictions saved to {predictions_file_path}")
