from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd 

class Sentiment:

    def __init__(self, data):
        self.data = data
        self.classifier = TextClassifier.load('en-sentiment')

    def value(self, paragraph):
        sentence = Sentence(paragraph)
        self.classifier.predict(sentence)
        if sentence.labels[0].value == 'POSITIVE':
            return sentence.labels[0].score
        elif sentence.labels[0].value == 'NEGATIVE':
            return - sentence.labels[0].score
    
    def extract_sentiments(self):
        df = pd.read_csv(self.data)
        sentiments1 = []
        sentiments2 = []
        arr1 = df['Paragraphs1'].tolist()
        arr2 = df['Paragraphs2'].tolist()

        for s in arr1:
            s = self.value(s)
            sentiments1.append(s)
            print(len(sentiments1))

        for s in arr2:
            s = self.value(s)
            sentiments2.append(s)
            print(len(sentiments2))

        df['Sentiments1'] = sentiments1
        df['Sentiments2'] = sentiments2


