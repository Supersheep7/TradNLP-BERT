import os
import numpy as np
import pandas as pd
import json
import re
import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

class Reddit_Parser():
    def __init__(self, train_path, eval_path, name):
        self.seed = 42
        self.name = name
        self.train_path = train_path
        self.eval_path = eval_path
        self.train_truth_table = pd.DataFrame(columns=['Paragraphs1', 'Paragraphs2', 'Truth_changes', 'file number'])
        self.eval_truth_table = pd.DataFrame(columns=['Paragraphs1', 'Paragraphs2', 'Truth_changes', 'file number'])
        self.train_single_sents = pd.DataFrame(columns=['Paragraphs', 'file number', 'F_vector', 'W_embeddings', 'Sentence_embedding'])
        self.eval_single_sents = pd.DataFrame(columns=['Paragraphs', 'file number', 'F_vector', 'W_embeddings', 'Sentence_embedding'])

    def openread(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                # Perform operations on the file
                content = file.read()
                
        except FileNotFoundError:
            print(f"The file '{filename}' does not exist.")
        except IOError:
            print(f"An error occurred while trying to read the file '{filename}'.")

        return content
    
    ''' Supervised '''

    def parse_split(self, problem_title, truth_title):
        
        # Open everything
        problem_paragraphs = self.openread(problem_title).split('\n')
        pairs = [(problem_paragraphs[i], problem_paragraphs[i + 1]) for i in range(len(problem_paragraphs) - 1)]
        paragraphs1 = [t[0] for t in pairs]
        paragraphs2 = [t[1] for t in pairs]

        truth_paragraphs = self.openread(truth_title)
        parsed = json.loads(truth_paragraphs)

        if len(pairs) is not len(parsed['changes']):            # TODO: solve this problem without going by hand
            print(len(pairs))
            print(len(parsed['changes']))
            print("Paragraph/Truth mismatch in file", problem_title) 

        return problem_paragraphs, paragraphs1, paragraphs2, parsed['changes']
        # Returns a truth table with parsed paragraphs on column 0 and truth values on column 1
    
    def get_data(self):

        # Files in train
        train_truth_table = self.train_truth_table
        train_files = os.listdir(self.train_path)
        train_problems = [os.path.join(self.train_path, file) for file in train_files if file.startswith('p')]
        train_truths = [os.path.join(self.train_path, file) for file in train_files if file.startswith('t')]
        entries = []
        for problem_title, truth_title in zip(train_problems, train_truths):
            paragraphs_for_sentences, paragraphs1, paragraphs2, truths = self.parse_split(problem_title, truth_title)
            for i in range(len(paragraphs1)):
                train_truth_table.loc[len(train_truth_table)] = [paragraphs1[i], paragraphs2[i], truths[i], ''.join(re.findall(r'\d+', problem_title[-10:]))]
            entries.append(paragraphs_for_sentences)
        entries = [item for sublist in entries for item in sublist]
        entries = list(set(entries))
        for i in range(len(entries)):
            self.train_single_sents.loc[len(self.train_single_sents), 'Paragraphs'] = entries[i]

        # Files in eval
        eval_truth_table = self.eval_truth_table
        eval_files = os.listdir(self.eval_path)
        eval_problems = [os.path.join(self.eval_path, file) for file in eval_files if file.startswith('p')]
        eval_truths = [os.path.join(self.eval_path, file) for file in eval_files if file.startswith('t')]
        for problem_title, truth_title in zip(eval_problems, eval_truths):
            paragraphs_for_sentences, paragraphs1, paragraphs2, truths = self.parse_split(problem_title, truth_title)
            for i in range(len(paragraphs1)):
                eval_truth_table.loc[len(eval_truth_table)] = [paragraphs1[i], paragraphs2[i], truths[i], ''.join(re.findall(r'\d+', problem_title[-10:]))]
            entries.append(paragraphs_for_sentences)
        entries = [item for sublist in entries for item in sublist]
        entries = list(set(entries))
        for i in range(len(entries)):
            self.eval_single_sents.loc[len(self.eval_single_sents), 'Paragraphs'] = entries[i]

    def get_table_pairwise(self, table='train', csv=False):
        
        if not csv:
            if table == 'train':
                return self.train_truth_table
            elif table == 'eval':
                return self.eval_truth_table
            else:
                ValueError('Missing an argument. The argument should be -train- or -eval-')

        else:
            self.train_truth_table.to_csv(f"csv/train-table-{self.name}.csv", index=False)
            self.eval_truth_table.to_csv(f"csv/eval-table-{self.name}.csv", index=False)

    ''' Unsupervised '''

    def sents_as_feature_vecs(self):

        if len(self.train_single_sents) == 0 and len(self.eval_single_sents) == 0:
            print("Call get_sents method before using this method")
            return
        paragraphs_train = self.train_single_sents['Paragraphs']
        paragraphs_eval = self.eval_single_sents['Paragraphs']
        for paragraph in paragraphs_train:
            self.feature_builder(paragraph)      # Builds the train_feature_table
        for paragraph in paragraphs_eval:
            self.feature_builder(paragraph)      # Builds the eval_feature_table

    def get_table_single(self, table='train', csv=False):
        
        if not csv:
            if table == 'train':
                return self.train_single_sents
            elif table == 'eval':
                return self.eval_single_sents
            else:
                ValueError('Missing an argument. The argument should be -train- or -eval-')

        else:
            self.train_single_sents.to_csv(f"csv/train-singles-{self.name}.csv", index=False)
            self.eval_single_sents.to_csv(f"csv/eval-singles-{self.name}.csv", index=False)
