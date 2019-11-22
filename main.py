import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random

#import json file in data variable
import json
with open('intents.json') as file:
    data = json.load(file)

#create empty lists for the data
words = []
labels = []
docs_x = []
docs_y = []

#loop for extracting data
for intentss in data['intents']:
    for patternss in intentss['patterns']:
        wordss = nltk.word_tokenize(patternss)
        words.extend(wordss)
        docs_x.append(wordss)
        docs_y.append(intentss["tag"])
    
    if intentss['tag'] not in labels:
        labels.append(intentss['tag'])

#stemming words (cut down words to its core)
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag =[]
    wordss = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wordss:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])]=1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
