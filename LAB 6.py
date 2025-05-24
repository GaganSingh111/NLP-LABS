from google.colab import drive
# Mount Google Drive to access files stored there
drive.mount('/content/drive')

# Install a specific version of librosa (audio processing library)
#pip install librosa==0.6.0

# Import necessary libraries
import numpy as np                   # For numerical operations
import matplotlib.pyplot as plt      # For plotting graphs
import matplotlib.style as ms        # For using predefined plot styles
import librosa.display               # For displaying audio features visually

# Set matplotlib style to 'seaborn-muted' for nicer looking plots
ms.use('seaborn-muted')

# Enable inline plotting for Jupyter/Colab notebooks
%matplotlib inline

import IPython.display as ipd        # For audio playback in notebooks
import librosa                       # Audio processing library

# Load an audio file from Google Drive; returns audio time series and sampling rate
data, sr = librosa.load('/content/drive/MyDrive/Datasets/Music.mp3')

# Print the shape of the audio data (number of samples)
print(data.shape)

# Compute a Mel-scaled spectrogram from the audio time series
S = librosa.feature.melspectrogram(data, sr)

# Convert power spectrogram (amplitude squared) to decibel (log) scale
log_S = librosa.power_to_db(S, ref=np.max)

# Plot the log-scaled Mel spectrogram as a line plot (less common; usually image is better)
plt.plot(log_S)
plt.show()

# Compute chromagram from the audio — this represents the energy of each pitch class (like notes C, C#, D, etc.)
chromagram = librosa.feature.chroma_stft(y=data, sr=sr)

# Create a figure with specific size to display the chromagram spectrogram
plt.figure(figsize=(15, 5))

# Display the chromagram with time on the x-axis and pitch class on the y-axis
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma')

# Calculate onset envelope (onset strength) from audio, which highlights likely note or beat beginnings
onset_env = librosa.onset.onset_strength(data, sr=sr)

# Estimate tempo (beats per minute) from the onset envelope
tempo = librosa.beat.tempo(onset_env, sr=sr)

# Use Harmonic-Percussive Source Separation (HPSS) to split audio into harmonic and percussive components
y_harmonic, y_percussive = librosa.effects.hpss(data)

# Track beats from the percussive component and estimate tempo and beat frames
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

# Print the estimated tempo in beats per minute
print(tempo)

# Print the array of beat frame indices detected in the audio
print(beats)
