import nltk
nltk.download("punkt")
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import pickle
import os
import pyttsx3
import speech_recognition as sr
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.models import load_model


stemmer = LancasterStemmer()

pkl_file = "data.pkl"

words = []
docs_x = []
docs_y = []
labels = []
stopwords = ['?', '!']

with open("intents.json", "rb") as intent_file:
  data = json.load(intent_file)

if os.path.isfile(pkl_file):
  with open(pkl_file, "rb") as saved_file:
    words, labels, training, out = pickle.load(saved_file)
else:
  for intent in data["intents"]:
    for pattern in intent["patterns"]:
      word_tokens = nltk.word_tokenize(pattern)
      words.extend(word_tokens)
      docs_x.append(word_tokens)
      docs_y.append(intent["tag"])

      if intent["tag"] not in labels:
        labels.append(intent["tag"])
  words = [stemmer.stem(word.lower()) for word in words if word != '?']
  words = sorted(list(set(words)))

  labels = sorted(labels)

  out_empty = [0 for _ in range(len(labels))]

  training = []
  out = []

  for x, doc in enumerate(docs_x):
    bag = []
    word_tokens = [stemmer.stem(word.lower()) for word in doc]
    for word in words:
      if word in word_tokens:
        bag.append(1)
      else:
        bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    out.append(output_row)
  training = np.array(training)
  out = np.array(out)

  with open(pkl_file, "wb") as save_file:
    pickle.dump((words, labels, training, out), save_file)

model = Sequential()
model.add(Dense(256, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(out[0]), activation='softmax'))

# opt = Adam(learning_rate=0.01)
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

if os.path.isfile("assistant_model.h5"):
  model = load_model("assistant_model.h5")
else:
  hist = model.fit(training, out, epochs = 1000, batch_size=8, verbose=1)
  model.save("assistant_model.h5")

def bag_of_words(sent, words):
  bag = [0] * len(words)
  sent_words = nltk.word_tokenize(sent)
  sent_words = [stemmer.stem(word.lower()) for word in sent_words]

  for se in sent_words:
    for i, w in enumerate(words):
      if w == se:
        bag[i] = 1
  return np.array(bag)

def listen():
  try:
    r = sr.Recognizer()
    with sr.Microphone() as source:
      print("Bot: Speak Now")
      audio = r.listen(source)
      query = r.recognize_google(audio)
      return query
  except Exception as ex:
    print("Bot: Sorry I can't understand")
    speak("Sorry I can't understand")
    return None

def speak(string):
  engine = pyttsx3.init()
  engine.say(string)
  engine.runAndWait()

def chat():
  start = "Hey there, how can I help you?"
  print(start)
  speak(start)

  while True:
    # inp = input("You: ")
    inp = listen()
    if inp is not None:
      print("You: " + inp)
      if inp == "quit":
        break
      results = model.predict(np.array([bag_of_words(inp, words)]))[0]
      res_index = np.argmax(results)
      tag = labels[res_index]
      responses = []
      for tg in data["intents"]:
        if tg["tag"] == tag:
          responses = tg["responses"]
      response = random.choice(responses)
      print("Bot: " + response)
      speak(response)

chat()