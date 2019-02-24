import speech_recognition as sr
import io
from google.cloud import speech_v1p1beta1 as speech
import os

def translate(audio_file):
    return translateGoogle(audio_file)

def translate_SR(audio_file):
    r = sr.Recognizer()

    file = sr.AudioFile(audio_file)
    with file as source:
        audio = r.record(source)

    text = r.recognize_google(audio)
    return text

def translateGoogle(audio_file):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="Creds.json"
    client = speech.SpeechClient()
    content = audio_file.read()
    audio = speech.types.RecognitionAudio(content=content)
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code='en-US',
        # Enable automatic punctuation
        enable_automatic_punctuation=True)

    response = client.recognize(config, audio)
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print('-' * 20)
        print('First alternative of result {}'.format(i))
        print('Transcript: {}'.format(alternative.transcript))
    return response.results