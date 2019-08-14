import numpy as np
import nltk
import os
from sklearn.externals import joblib

NUM_Feature = 50

if os.path.exists('glove.pkl'):
    w2v = joblib.load('glove.pkl')
else:
    w2v = loadGloveModel("./glove.6B/glove.6B.50d.txt")

if os.path.exists('obscureWords.pkl'):
    obscure = joblib.load('obscureWords.pkl')
else:
    obscure = {}

def main():
    #save_embedded_doc(docEncoder("./docs&Qs/1/doc.txt"))
    q = queryEncoder("How does a spitting cobra use its spit to protect itself?")
    save_embedded_query(q)

def docEncoder(text):
    global obscure
    text = text.lower()
    sents = nltk.sent_tokenize(text)
    word = []
    for sent in sents:
        word.append(nltk.word_tokenize(sent))
    doc_embed = []
    for i in range(len(word)):
        sent_embed = []
        for j in range(len(word[i])):
            # i: sentence number
            # j: word number in sentence
            try:
                sent_embed.append(w2v[word[i][j]])
            except:
                try:
                    sent_embed.append(obscure[word[i][j]])
                except:
                    vec = np.random.rand(NUM_Feature)*2 - 1
                    sent_embed.append(vec)
                    obscure[word[i][j]] = vec
        doc_embed.append(np.array(sent_embed))
    joblib.dump(obscure,'obscureWords.pkl')
    return doc_embed

def queryEncoder(query):
    global obscure
    query = query.lower()
    text = nltk.word_tokenize(query)
    query_embed = []
    for i in range(len(text)):
        try:
            query_embed.append(w2v[text[i]])
        except:
            try:
                query_embed.append(obscure[text[i]])
            except:
                vec = np.random.rand(NUM_Feature)*2 - 1
                query_embed.append(vec)
                obscure[text[i]] = vec
    query_embed = np.array(query_embed)
    joblib.dump(obscure,'obscureWords.pkl')
    return query_embed

def test_set_load():
    pass

def createGlovedict():
    w2v = loadGloveModel("./glove.6B/glove.6B.50d.txt")
    joblib.dump(w2v,'glove.pkl')

def save_embedded_doc(doc,name = 'text_emb_doc.pkl'):
    joblib.dump(doc,name)
def save_embedded_query(query,name = "query_emb.pkl"):
    joblib.dump(query,name)

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile,'r',encoding='utf-8') as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
    return model

if __name__ == "__main__":
    main()