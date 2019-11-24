import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import pickle

#import json file in data variable
import json
with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words,labels, training, output = pickle.load(f)
except:
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

    #create bag of words list
    for x, doc in enumerate(docs_x):
        bag =[]
        wordss = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            #check if there is a word, if it is then add 1
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

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels, training, output),f)

#Part 2 : The DNN using tensorflow
#set initial graph
#tf2 ver
#tensorflow.compat.v1.reset_default_graph()
#tf1 ver
tensorflow.reset_default_graph()
#input data
net = tflearn.input_data(shape=[None, len(training[0])])
#hidden layer 8 node
net = tflearn.fully_connected(net, 8)
#hidden layer 8 node
net = tflearn.fully_connected(net, 8)
#output layer using softmax (higher probability output)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    #epoch is number of data trained by the model
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w ==se:
                bag[i] = 1
    
    return numpy.array(bag)

def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])
        print(results)

chat()
