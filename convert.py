from datetime import datetime
import connexion
from TextSummarizer import summarize,summarize_text
import json


def audio():
    files =connexion.request.files
    summary = summarize(files[''])
    return summary



def text():
    files = connexion.request.json

    text = files['textData']
    return summarize_text(text)