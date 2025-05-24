import speech_recognition as sr  # Library for speech recognition
import pyttsx3                 # Library for text-to-speech conversion

# Initialize the recognizer object
r = sr.Recognizer()

# Function to convert given text to speech
def SpeakText(command):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    # Queue the command (text) to be spoken
    engine.say(command)
    # Process and play the speech
    engine.runAndWait()

# Infinite loop to keep listening to user speech
while True:
    try:
        # Use the default microphone as the audio source
        with sr.Microphone() as source2:
            # Adjust for ambient noise to calibrate the recognizer
            # duration=0.2 seconds means it listens 0.2s to adjust noise threshold
            r.adjust_for_ambient_noise(source2, duration=0.2)

            # Listen to the source and store audio data in audio2 variable
            audio2 = r.listen(source2)

            # Recognize speech using Google Web Speech API
            MyText = r.recognize_google(audio2)
            # Convert the recognized text to lowercase for uniformity
            MyText = MyText.lower()

            # Print what was heard
            print("Did you say " + MyText)
            # Convert recognized text back to speech and speak it
            SpeakText(MyText)

    # Catch network/request errors with the Google API
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    # Catch errors when speech was unintelligible
    except sr.UnknownValueError:
        print("unknown error occured")
