# Import necessary libraries for data manipulation, text processing, and modeling
import pandas as pd  # Data handling for tabular data
import re  # Regular expressions for text cleaning tasks
import gensim  # Library for topic modeling and word embeddings
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords  # List of common stopwords like "the", "is"
from nltk.stem import WordNetLemmatizer  # To convert words to their base form (lemmatization)
from string import punctuation  # Contains punctuation characters to remove
from gensim.corpora import Dictionary  # Create mapping of words to integer IDs for LDA
from nltk.tokenize import word_tokenize  # To split text into individual words (tokens)
from gensim.models.ldamodel import LdaModel, CoherenceModel  # LDA model and evaluation methods
import pyLDAvis  # Library for interactive visualization of topic models
import pyLDAvis.gensim  # Gensim integration with pyLDAvis
import matplotlib.pyplot as plt  # For plotting (optional)
%matplotlib inline  # To show matplotlib plots inline in Jupyter notebooks

# Load the dataset from a JSON URL containing text documents from 20 newsgroups
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
print("Dataset Preview:")
print(df.head())  # Show first 5 rows to understand the data structure

# Function to remove email addresses using regex pattern matching
def removing_email(text):
    # This pattern matches anything like 'user@domain.com' and removes it
    return re.sub(r'\S*@\S*\s?', ' ', text)

# Function to keep only letters and spaces, removing numbers and special chars
def only_words(text):
    # Replace anything not A-Z or a-z or whitespace with a space
    return re.sub(r'[^a-zA-Z\s]', ' ', text)

# Define stopwords that we want to exclude from analysis to reduce noise
stop_words = (
    list(set(stopwords.words('english'))) +  # Common English stopwords
    list(punctuation) +  # Punctuation marks such as '.', ',', '!'
    ['\n', '----', '---\n\n\n\n\n']  # Additional noise tokens found in data
)

# Initialize the lemmatizer object for converting words to base form
lem = WordNetLemmatizer()

# Comprehensive cleaning function to prepare text for modeling
def cleaning(text):
    text = text.lower()  # Lowercase all text to ensure uniformity
    words = word_tokenize(text)  # Split text into individual words/tokens
    # Remove stopwords and punctuation, keep words longer than 2 chars
    words = [w for w in words if w not in stop_words and len(w) >= 3]
    # Lemmatize words considering verbs to unify different word forms (e.g., "running" → "run")
    lemma = [lem.lemmatize(w, 'v') for w in words]
    return lemma

# Apply email removal, non-alpha filtering, and cleaning sequentially on text column
df['without email'] = df['content'].apply(removing_email)  # Step 1: Remove emails
df['only words'] = df['without email'].apply(only_words)  # Step 2: Remove non-letters
df['clean content'] = df['only words'].apply(cleaning)  # Step 3: Clean & tokenize

print("\nProcessed Data Preview:")
print(df.head())  # Show cleaned data preview for validation

# Convert cleaned documents into a list of tokenized lists for gensim processing
clean_doc = list(df['clean content'].values)

# Create a dictionary mapping from word to unique ID based on the entire corpus
dictionary = Dictionary(clean_doc)
# Optional: Filter out words that are too rare or too common to improve model quality
# dictionary.filter_extremes(no_below=5, no_above=0.5)

# Convert each document into Bag-of-Words format (list of (word_id, frequency) tuples)
corpus = [dictionary.doc2bow(doc) for doc in clean_doc]

# Train an LDA topic model with 5 topics
ldamodel = LdaModel(
    corpus=corpus,  # Input corpus in BoW format
    id2word=dictionary,  # Mapping from id to word
    num_topics=5,  # Number of latent topics to extract
    random_state=42,  # Fix random seed for reproducibility
    update_every=1,  # Update model every pass
    passes=50,  # Number of iterations over corpus for better convergence
    chunksize=100,  # Number of documents processed at once
    alpha='auto',  # Automatically learn document-topic density
    eta='auto'  # Automatically learn topic-word density
)

# Display the top 10 words in each of the 5 discovered topics with their weights
print("\nDiscovered Topics:")
for idx, topic in ldamodel.print_topics(num_words=10):
    print(f"Topic {idx + 1}: {topic}")

# Evaluate model using log perplexity (lower is better)
print("\nLog Perplexity:", ldamodel.log_perplexity(corpus))

# Calculate coherence score (c_v) to measure semantic similarity within topics (higher is better)
coherence_cv = CoherenceModel(
    model=ldamodel,
    texts=clean_doc,
    dictionary=dictionary,
    coherence='c_v'
)
print("Coherence (c_v):", coherence_cv.get_coherence())

# Calculate coherence score (u_mass), an alternative metric (closer to 0 is better)
coherence_umass = CoherenceModel(
    model=ldamodel,
    texts=clean_doc,
    dictionary=dictionary,
    coherence='u_mass'
)
print("Coherence (u_mass):", coherence_umass.get_coherence())

# Prepare an interactive visualization for the topics using pyLDAvis
pyLDAvis.enable_notebook()  # Enable visualization display in Jupyter
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)  # Prepare visualization data
vis  # Render the interactive visualization inline
