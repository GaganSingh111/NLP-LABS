# Import Google Colab drive module to mount Google Drive
from google.colab import drive

# Mount Google Drive to access files stored there at '/content/drive'
drive.mount('/content/drive')

# Import audio processing libraries
import librosa  # Library for audio and music analysis
import soundfile  # To read/write sound files
import os, glob, pickle  # For file handling and pattern matching
import numpy as np  # For numerical operations

# Import scikit-learn modules for model building and evaluation
from sklearn.model_selection import train_test_split  # To split dataset into train/test sets
from sklearn.neural_network import MLPClassifier  # Multi-layer perceptron classifier
from sklearn.metrics import accuracy_score  # To calculate classification accuracy

# Function to extract audio features from a sound file
# Parameters:
# - file_name: path to the audio file
# - mfcc: boolean, whether to extract MFCC features
# - chroma: boolean, whether to extract chroma features
# - mel: boolean, whether to extract Mel spectrogram features
def extract_feature(file_name, mfcc, chroma, mel):
    # Open the sound file in read mode
    with soundfile.SoundFile(file_name) as sound_file:
        # Read the entire audio signal as a float32 numpy array
        X = sound_file.read(dtype="float32")
        # Get the sampling rate of the audio file
        sample_rate = sound_file.samplerate
        
        # Initialize the result array to store features
        result = np.array([])
        
        # If chroma features are requested, calculate the Short-Time Fourier Transform (STFT) magnitude
        if chroma:
            stft = np.abs(librosa.stft(X))
        
        # Extract MFCC features if requested
        if mfcc:
            # Compute 40 MFCCs and take the mean over time (axis=0)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            # Append MFCC features to the result array
            result = np.hstack((result, mfccs))
        
        # Extract chroma features if requested
        if chroma:
            # Compute chroma features from the STFT magnitude and take mean over time
            chroma_feature = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            # Append chroma features to the result array
            result = np.hstack((result, chroma_feature))
        
        # Extract Mel spectrogram features if requested
        if mel:
            # Compute Mel spectrogram and take mean over time
            mel_feature = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            # Append Mel features to the result array
            result = np.hstack((result, mel_feature))
    
    # Return the concatenated feature vector
    return result

# Dictionary mapping dataset emotion codes to their descriptive labels
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# List of emotions we want to focus on (subset of all emotions)
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Function to load the audio data, extract features, and prepare train/test splits
# Parameters:
# - test_size: fraction of data reserved for testing (default 0.2 = 20%)
def load_data(test_size=0.2):
    x, y = [], []  # Lists to store features (x) and labels (y)
    
    # Iterate over all .wav files in the specified dataset directory
    for file in glob.glob("/content/drive/My Drive/Datasets/Voice Samples/Actor_*/*.wav"):
        # Extract the filename from the full path
        file_name = os.path.basename(file)
        
        # Extract the emotion code from the filename (3rd element when split by '-')
        emotion_code = file_name.split("-")[2]
        # Map the emotion code to the descriptive label
        emotion = emotions[emotion_code]
        
        # Only process files with emotions we want to observe
        if emotion not in observed_emotions:
            continue  # Skip this file if emotion not in the list
        
        # Extract audio features from the file (MFCC + Chroma + Mel)
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        
        # Append the extracted features and label to the lists
        x.append(feature)
        y.append(emotion)
    
    # Split the dataset into training and testing sets and return them
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Load the dataset and split into train and test sets with 25% test size
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# Print the number of training and testing samples
print((x_train.shape[0], x_test.shape[0]))

# Print the total number of features extracted per audio sample
print(f'Features extracted: {x_train.shape[1]}')

# Initialize a Multi-layer Perceptron classifier with the following parameters:
# alpha: L2 penalty (regularization term)
# batch_size: mini-batch size for training
# epsilon: term for numerical stability in optimizer
# hidden_layer_sizes: one hidden layer with 300 neurons
# learning_rate: adaptive learning rate adjustment
# max_iter: maximum number of iterations for training
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the MLP model using the training data
model.fit(x_train, y_train)

# Use the trained model to predict emotions for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy by comparing predicted labels with true test labels
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy as a percentage with two decimal places
print("Accuracy: {:.2f}%".format(accuracy * 100))
