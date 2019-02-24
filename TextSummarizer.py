import numpy as np
import pandas as pd
#nltk.download('punkt') # one time execution
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from SpeechToText import translate
import re



def remove_stopwords(sen,stop_words):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def summarize(audio_file):
    #df = pd.read_csv("tennis_articles_v4.csv")
    #text = 'Female Speaker 1 Good morning and welcome to day two of the Annual Banking and Financial Services Conference. Its my pleasure to have all of you here. Why dont I start off by maybe asking both of you, Johnny and Selena. People on the sell side like me and investors have been focused on this whole operating leverage thing for such a long time. At the same time, we certainly want to make sure that the stuff that youre eliminating is inefficiency as opposed to investment for growth. What was the process that you went through in order to make sure that what you were eliminating was a waste as opposed to the real growth spending?  Male Speaker 1 We are actually doing two things, Nicole, and working on the two dimensions at the same time. First of all, expense management. Expense management is more than just expense management. It should be cost management because it also includes defining what is the right level and number of people that you need in any operation. So, we look at head count as well. Thats what we call expense management, and that should yield results in the short term. The second one is through reengineering, and even more than that, business transformation. For that, we have the possibility of doing it now that we are moving to a global operating platform based on unique IT platform for the core systems, for internet, for mobile. Everything is going to become global, and if you have one system, this is the power of one. You can actually save a lot of money from having different software groups in different countries developing and adjusting systems. You have to have of course a very robust global governance system. We have created that over the last year through the Global Consumer Council. So, both expense management and reengineering. Now, the key question is, what is there for our people? How do we actually motivate them to actually focus on expense management and reengineering? By basically promising that any savings that we have from our current cost base will be reinvested in revenue-generating opportunities. So, its creating the discipline from within the company and changing the culture so that all are motivated to really go and do their best in terms of operating efficiency and business transformation so that they can have more resources to continue to invest in what they deem to be revenue-generating activities. Those projects, those initiatives, are of course considered at the global level. This is one governance process because we want to make sure that we invest in those opportunities that offer the best returns on Basel III, going forward quite clearly and generate the best earning role for our shareholders. Female Speaker 2 The same process that Johnny describes for consumer, we extend to the entire firm. Our approach is every business is expected to generate somewhere between 3-5% efficiency saves in their expense base every year, and we take that into the budget. To the extent that we are able to deliver those efficiency saves, theres more dollars to invest. Now, recognizing, coming out of 2008-2009 where we had significantly cutback and part of what drove the negative operating leverage that you saw during the latter part of 2010 and the early part of 2011, we decided to reinvest in advance of generating all those expense saves. If you look at some of the material that weve put out this year through the third quarter for the full firm, we would have had incremental investment spending of about 2.8 billion USD, which has been about half paid by those efficiency saves. So, weve already taken out about 1.4 billion USD of the expense base through the third quarter of this year, full firm. So, weve been in a mode of invest first, get the expense saves, and see where the revenue comes. Now that weve made the initial investment in dollars, weve moved much more onto the scenario that Johnny laid out, where in the future it will be more, how much you do you save becomes how much of invested dollars you actually have for the firm, and thats whats generating then that moved back towards that positive operating leverage. Female Speaker 1 Right. Thanks Selena.'
    #sentences = re.findall(r'(?:\d[,.]|[^,.])*(?:[,.]|$)', text)
    #print(len(sentences))
    #for s in df['article_text']:
      #sentences.append(sent_tokenize(s))
    #sentences = [y for x in sentences for y in x] # flatten list
    text = str(translate(audio_file))
    sentences = re.findall(r'(?:\d[,.]|[^,.])*(?:[,.]|$)', text)
    return summarize_sentences(sentences)


def summarize_text(text):
    sentences = re.findall(r'(?:\d[,.]|[^,.])*(?:[,.]|$)', text)
    return summarize_sentences(sentences)



def summarize_sentences(sentences):
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()


    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    stop_words = stopwords.words('english')
    clean_sentences = [remove_stopwords(r.split(), stop_words) for r in clean_sentences]


    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)


    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = \
                cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]


    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)


    summarized_text = []
    # Extract top 10 sentences as the summary
    topwords = []
    lowest = int(len(sentences) / 3)

    if lowest == 0:
        lowest = len(sentences)
    for i in range(lowest):
        topwords.append(ranked_sentences[i][1])

    for j in range(len(sentences)):
        if sentences[j] in topwords:
            summarized_text.append(sentences[j])


    return ' '.join(summarized_text)
