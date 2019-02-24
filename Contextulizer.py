from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk import pos_tag
import re

def contextualize(text):
    trigger_words = ['meet', 'catch-up', 'schedule', 'arrange', 'see-you-at', 'see-you-tommorrow']
    for trigger in trigger_words:
        syns = wordnet.synsets(trigger)
        if len(syns) > 0:
            trigger_words.append(syns[0].name())





    tokenized_text=re.findall(r'(?:\d[,.]|[^,.])*(?:[,.]|$)', text)
    result = []

    for sentence in tokenized_text:
        for trigger in trigger_words:
            if trigger in sentence:
                result.append(sentence)
                break



    ps = PorterStemmer()

    filtered_stemmed = []

    for i, sent in enumerate(result):
        flag = True
        x = pos_tag(sent.split())
        for tag in x:
            if  tag[1] == "VBD" or tag[1] == "VBN":
                flag = False
                break
        if(flag):
            filtered_stemmed.append(ps.stem(result[i]))


    return filtered_stemmed
