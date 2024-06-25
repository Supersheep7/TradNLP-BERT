from sentence_transformers import SentenceTransformer
import pandas as pd

class S_Bert():
        
    def __init__(self, data, path):
            self.data = data
            self.sbert = SentenceTransformer(path)

    def extract_s_embeddings(self):
    
        sbert = self.sbert
    
        df = pd.read_csv(self.data)
        embeddings1 = []
        embeddings2 = []
        arr1 = df['Paragraphs1'].tolist()
        arr2 = df['Paragraphs2'].tolist()

        for s in arr1:
            s = sbert.encode(s)
            embeddings1.append(s)
            print(len(embeddings1))

        for s in arr2:
            s = sbert.encode(s)
            embeddings2.append(s)
            print(len(embeddings2))

        df['S_embeddings1'] = embeddings1
        df['S_embeddings2'] = embeddings2