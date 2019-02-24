import speech_recognition as sr

def translate(audio_file):

    r = sr.Recognizer()

    file = sr.AudioFile(audio_file)
    with file as source:
        audio = r.record(source)

    text = r.recognize_google(audio)
    return text
