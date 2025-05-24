# Mount Google Drive to access files stored there
from google.colab import drive
# Mounting drive at '/content/drive' allows you to read files from your Google Drive folder in Colab
drive.mount('/content/drive')

# Import pandas for data manipulation and analysis
import pandas as pd  

# Import necessary TensorFlow Keras modules for building the neural network
from tensorflow.keras.models import Sequential  # Model container to stack layers
from tensorflow.keras.layers import LSTM, Dense, Embedding  # Different layers: LSTM, fully connected, and embedding
from tensorflow.keras.preprocessing.text import Tokenizer  # To convert text into sequences of integers
from tensorflow.keras.preprocessing.sequence import pad_sequences  # To pad sequences to the same length

# Import NLTK modules for text preprocessing
from nltk.tokenize import word_tokenize  # To split sentences into words (tokens)
from nltk.stem import WordNetLemmatizer  # To convert words to their base/dictionary form
from nltk.corpus import stopwords  # Common words to remove (e.g., 'the', 'is')
from string import punctuation  # List of punctuation symbols to remove

# Import numpy for numerical operations and tqdm for progress bars
import numpy as np  
from tqdm import tqdm  
tqdm.pandas()  # Enable progress bar for pandas' apply function

"""
Dataset contains Quora questions with labels:
- qid: question id (not used in modeling)
- question_text: actual text of the question (input)
- target: 0 for sincere, 1 for insincere (output label)
"""
# Load the dataset CSV file from Google Drive path into a DataFrame
df = pd.read_csv('/content/drive/MyDrive/Datasets/Quora Text Classification Data.csv')
df.head()  # Show first 5 rows to understand data structure

# Download necessary NLTK datasets for stopwords, tokenizer models, and lemmatization
import nltk
nltk.download('stopwords')  # List of stopwords to remove
nltk.download('punkt')  # Tokenizer pre-trained model to split sentences into words
nltk.download('wordnet')  # WordNet corpus needed for lemmatizer

# Combine English stopwords with punctuation characters to remove both from text
stop_words = stopwords.words('english') + list(punctuation)

# Initialize the WordNet lemmatizer object that converts words to their root/base form
lem = WordNetLemmatizer()

def cleaning(text):
    """
    Text preprocessing pipeline to clean input text:
    1. Convert text to lowercase to standardize (e.g., 'Python' -> 'python')
    2. Tokenize text into individual words (tokens)
    3. Remove stopwords and punctuation tokens (words like 'the', ',', 'is')
    4. Lemmatize each token to reduce to base form (e.g., 'running' -> 'run')
    5. Join tokens back into a cleaned string
    """
    text = text.lower()  # Normalize case for uniformity
    words = word_tokenize(text)  # Split into words/tokens
    words = [w for w in words if w not in stop_words]  # Remove stopwords & punctuation tokens
    words = [lem.lemmatize(w) for w in words]  # Lemmatize each word to base form
    return ' '.join(words)  # Recombine cleaned tokens into a single string

# Apply the cleaning function to each question_text with a progress bar
df['Clean Text'] = df['question_text'].progress_apply(cleaning)

# Unzip the pre-trained GloVe embeddings file (42 billion tokens, 300-dim vectors)
#unzip '/content/drive/MyDrive/Word Embeddings/glove.42B.300d.zip'

# Initialize a dictionary to store word-to-vector mappings from GloVe
embedding_values = {}

# Open GloVe file which contains lines like: "word 0.123 0.532 ... 0.045" (300 floats)
f = open('/content/glove.42B.300d.txt', encoding='utf-8')

# Read each line, split by space:
# - first element is the word (string)
# - rest are 300 floating point numbers representing the word's semantic vector
for line in tqdm(f):
    value = line.split(' ')  # Split line into list: [word, float1, float2, ..., float300]
    word = value[0]  # Extract the word itself (string)
    coef = np.array(value[1:], dtype="float32")  # Convert remaining 300 values to numpy float array
    if coef is not None:
        embedding_values[word] = coef  # Save mapping: word -> 300-dim vector (numpy array)

# Initialize tokenizer again to convert cleaned text into sequences of integers
tokenizer = Tokenizer()

# 'x' is the cleaned question text column
x = df['Clean Text']

# 'y' is the target label column (0 or 1)
y = df['target']

# Build the word index dictionary by scanning all cleaned texts:
# Assigns unique integer ID to each unique word in dataset vocabulary
tokenizer.fit_on_texts(x)

# Convert each cleaned text sentence to a sequence of integers
# Each integer corresponds to a word's index in tokenizer.word_index
seq = tokenizer.texts_to_sequences(x)

# Pad/truncate sequences to fixed length of 300 tokens so inputs are uniform size
# Short sequences get 0-padded at start; long sequences truncated from the start
pad_seq = pad_sequences(seq, maxlen=300)

# Calculate vocabulary size (number of unique words + 1 for padding token at index 0)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Create embedding matrix of shape (vocab_size, 300)
# Each row corresponds to a word index; each column corresponds to a GloVe embedding dimension
embedding_matrix = np.zeros((vocab_size, 300))  # Initialize all zeros for words not in GloVe

# For each word and its index in tokenizer vocabulary:
for word, i in tqdm(tokenizer.word_index.items()):
    value = embedding_values.get(word)  # Lookup GloVe vector (300 floats) for word
    if value is not None:
        embedding_matrix[i] = value  # Assign GloVe vector to embedding matrix row corresponding to word index

"""
Model architecture:
- Embedding Layer: Converts word indices into 300-dimensional vectors using embedding_matrix
  (weights frozen so pre-trained GloVe vectors are not updated during training)
- LSTM Layer: Processes sequence of embeddings and learns context (50 units)
- Dense Layer: Fully connected layer with 128 ReLU units to learn complex patterns
- Output Layer: Single neuron with sigmoid activation to output probability of class 1 (insincere)
"""
model = Sequential()

# Embedding layer takes integer word indices as input and outputs their GloVe embeddings
model.add(Embedding(vocab_size, 300, input_length=300,
                    weights=[embedding_matrix], trainable=False))

# LSTM layer to capture sequential dependencies, returns final output vector
model.add(LSTM(50, return_sequences=False))

# Dense hidden layer with ReLU activation to model non-linear features
model.add(Dense(128, activation='relu'))

# Output layer with sigmoid activation for binary classification (0 or 1)
model.add(Dense(1, activation='sigmoid'))

# Compile model with Adam optimizer, binary crossentropy loss for classification, and accuracy metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on padded sequences and labels, use 20% data as validation set, for 5 epochs
history = model.fit(pad_seq, y, validation_split=0.2, epochs=5)
