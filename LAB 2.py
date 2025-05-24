# Import gensim's model downloader utility
import gensim.downloader

# Suppress warnings for clean output (useful in presentation, but not during debugging)
import warnings
warnings.filterwarnings('ignore')

"""
Gensim Model Downloader Explanation:
- gensim.downloader provides access to various pre-trained word embedding models
- These models encode semantic meaning of words into dense vector representations
- Examples: Word2Vec, GloVe, FastText, etc.
"""

# List all available pre-trained models to choose from
print("Available pre-trained models in gensim:")
print(list(gensim.downloader.info()['models'].keys()))

"""
GloVe Model Loading:
- 'glove-wiki-gigaword-300' is a GloVe model trained on Wikipedia + Gigaword corpus
- Produces 300-dimensional word vectors
- Trained using matrix factorization on word co-occurrence statistics
- Captures both syntactic and semantic relationships
"""
print("\nLoading glove-wiki-gigaword-300 model (this may take a few minutes)...")
glove_model = gensim.downloader.load('glove-wiki-gigaword-300')

"""
Most Similar Words Demonstration:
- most_similar() retrieves words with the highest cosine similarity to the input word
- Helps evaluate how well the model captures semantic relationships
"""
print("\nWords most similar to 'technology':")
print(glove_model.most_similar('technology'))

"""
Expected Output:
- Returns related terms such as 'technologies', 'innovation', 'engineering'
- Shows model's understanding of the domain
"""

print("\nWords most similar to 'Science':")
print(glove_model.most_similar('Science'))

"""
Case Sensitivity:
- GloVe embeddings are case-sensitive
- 'Science' and 'science' may give slightly different results
"""

print("\nWords most similar to 'arts':")
print(glove_model.most_similar('arts'))

"""
Semantic Clustering:
- Expected to return terms from culture, literature, humanities, etc.
- Demonstrates how well the model understands different contexts
"""

"""
Word Similarity Measurement:
- similarity() gives cosine similarity score between two word vectors
- Score ranges from -1 (opposite) to +1 (identical meaning)
"""
print("\nSimilarity between 'hot' and 'cold':")
print(glove_model.similarity('hot', 'cold'))

"""
Interpretation:
- Even though 'hot' and 'cold' are opposites, similarity might still be relatively high
- Reason: Appear in similar contexts (e.g., weather, temperature)
- Word vectors are based on distributional semantics: "You shall know a word by the company it keeps"
"""

"""
Further Concepts:

1. What Makes GloVe Unique:
- Combines benefits of matrix factorization and local context-based methods
- Focuses on global co-occurrence statistics
- Embeddings are pre-computed and not updated dynamically

2. Applications of GloVe Embeddings:
- Input features for deep learning models
- Text classification, sentiment analysis
- Clustering and topic modeling
- Information retrieval systems

3. Limitations:
- Fixed vocabulary (OOV words can’t be embedded)
- No dynamic context (same vector for all usages of a word)
- No handling of multi-word expressions directly

4. Comparison to Other Models:
- Word2Vec: Context prediction-based (CBOW or Skip-gram)
- GloVe: Global co-occurrence-based
- FastText: Handles subword information, good for rare/unknown words
- BERT/ELMo: Contextual embeddings, dynamically generated per sentence
"""
