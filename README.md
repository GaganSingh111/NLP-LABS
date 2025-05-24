# Project Labs Overview

## Summary

This repository contains 9 lab projects focusing on machine learning, deep learning, audio processing, natural language processing, and neural network architectures. These labs demonstrate practical implementations including:

- Audio feature extraction and speech emotion recognition
- Sequence-to-sequence models for dialogue generation
- Deep learning models using Keras and scikit-learn
- Text tokenization and preprocessing techniques
- Neural network training, evaluation, and prediction workflows

---

## Labs Included

1. **Speech Emotion Recognition with Audio Features**  
   Extract MFCC, Chroma, and Mel features from audio files and classify emotions using an MLP classifier.

2. **Google Drive Integration for Dataset Access**  
   Mount Google Drive in Google Colab to load datasets for training models.

3. **Text Preprocessing and Tokenization**  
   Tokenize and sequence text data from movie dialogues with padding and truncation.

4. **Sequence-to-Sequence (Seq2Seq) Model for Dialogue Generation**  
   Build and train a bidirectional LSTM encoder-decoder model to generate conversational text.

5. **One-Hot Encoding of Target Sequences**  
   Prepare categorical output data for training neural networks in text generation tasks.

6. **Training and Evaluation of Deep Learning Models**  
   Train models with categorical cross-entropy loss, evaluate accuracy, and save predictions.

7. **Data Loading and Splitting**  
   Use scikit-learn to split datasets into training and testing subsets.

8. **Saving Predictions with Timestamped Filenames**  
   Export generated outputs in JSON format with clear actual vs. generated text comparisons.

9. **Usage of Keras Functional API**  
   Define complex neural network architectures including embedding, LSTM, repeat vector, and time-distributed dense layers.

---

## Setup Instructions

- Use Python 3.x with libraries: `keras`, `tensorflow`, `librosa`, `soundfile`, `numpy`, `scikit-learn`
- For audio-related labs, install `librosa` and `soundfile`
- Google Colab is recommended for GPU acceleration and Google Drive dataset access
- Mount Google Drive to access datasets located in `/content/drive/MyDrive/`

---

## Running the Labs

1. Clone the repository or upload files to Google Colab.
2. Mount Google Drive for dataset access if needed.
3. Install necessary packages using pip (`!pip install librosa soundfile tensorflow keras scikit-learn`).
4. Run each lab script/notebook step-by-step.
5. Monitor outputs including model accuracy, training logs, and saved prediction files.

---

## Notes

- Modify dataset paths as needed to your local or cloud environment.
- Hyperparameters (like vocabulary size, embedding dimensions, batch size) can be tuned per lab.
- For best results, train on GPU-enabled environments.
- Extend or adapt models for new datasets or tasks as desired.

---

## Contact

For support or collaboration, please contact:

**Gagan**  
**Email: gagan.official001@gmail.com**

