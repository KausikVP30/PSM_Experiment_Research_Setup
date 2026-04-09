from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, documents):
        embeddings = self.model.encode(documents, convert_to_numpy = True)
        return embeddings

    def encode_query(self, query):
        embeddings = self.model.encode([query], convert_to_numpy = True)
        return embeddings

    