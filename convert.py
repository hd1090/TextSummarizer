from datetime import datetime
import connexion
from TextSummarizer import summarize,summarize_text
from Contextulizer import contextualize
import json


def audio():
    files =connexion.request.files
    summary = summarize(files['upfile'])
    return summary



def text():
    files = connexion.request.json

    text = files['textData']
    return summarize_text(text)



def context():
    files = connexion.request.json
    text = files['textData']
    return contextualize(text)