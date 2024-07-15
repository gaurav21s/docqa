import json
import os
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from utils.logger import logger

class DataHandler:
    def __init__(self, collection_name: str = "nvidia_cuda_docs"):
        """
        Initialize the DataHandler with a specified collection name.

        Args:
            collection_name (str): Name of the Milvus collection to use or create.
        """
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.connect_to_milvus()
        self.create_collection_if_not_exists()
        self.collection = Collection(self.collection_name)

    def connect_to_milvus(self):
        logger.info("Milvus collection connected")
        connections.connect("default", host="localhost", port="19530")

    def create_collection_if_not_exists(self):
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
            ]
            schema = CollectionSchema(fields, "NVIDIA CUDA Documentation Chunks")
            Collection(self.collection_name, schema)
            # print(f"Collection '{self.collection_name}' created.")
            logger.info(f"Collection '{self.collection_name}' created.")
        else:
            # print(f"Collection '{self.collection_name}' already exists.")
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def load_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r') as f:
            return json.load(f)

    def preprocess_text(self, text: str) -> str:
        return text.lower()

    def advanced_chunking(self, documents: List[Dict], chunk_size: int = 200, overlap: int = 50) -> List[Tuple[str, str, str]]:
        """
        Perform advanced chunking on the input documents.

        This method includes sentence splitting, overlap handling, similarity checking,
        TF-IDF vectorization, and LDA topic modeling for refined chunking.

        Args:
            documents (List[Dict]): List of input documents.
            chunk_size (int): Size of each chunk in sentences.
            overlap (int): Number of overlapping sentences between chunks.

        Returns:
            List[Tuple[str, str, str]]: List of chunks as (URL, content, title) tuples.
        """
        logger.info("Chunking and refining the data")
        chunks = []
        for doc in documents:
            text = self.preprocess_text(doc['content'])
            sentences = text.split('.')
            for i in range(0, len(sentences), chunk_size - overlap):
                chunk = ' '.join(sentences[i:i+chunk_size])
                if chunks and i > 0:
                    prev_chunk = chunks[-1][1]
                    similarity = self.model.encode([chunk, prev_chunk]).mean()
                    if similarity > 0.8:
                        chunks[-1] = (doc['url'], chunks[-1][1] + ' ' + chunk, chunks[-1][2])
                        continue
                chunks.append((doc['url'], chunk, doc['title']))
        
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf = vectorizer.fit_transform([c[1] for c in chunks])
        
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        topic_distribution = lda.fit_transform(tfidf)
        
        refined_chunks = []
        for i, chunk in enumerate(chunks):
            main_topic = np.argmax(topic_distribution[i])
            if topic_distribution[i][main_topic] > 0.3:
                refined_chunks.append(chunk)
        
        return refined_chunks

    def create_embeddings(self, chunks: List[Tuple[str, str, str]]) -> np.ndarray:
        return self.model.encode([c[1] for c in chunks])

    def setup_milvus(self):
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()

    def store_in_milvus(self, embeddings: np.ndarray, chunks: List[Tuple[str, str, str]]):
        entities = [
            {
                'embedding': embedding.tolist(),
                'url': chunk[0],
                'title': chunk[2],
                'content': chunk[1]
            }
            for embedding, chunk in zip(embeddings, chunks)
        ]
        self.collection.insert(entities)
        self.collection.flush()

    def process_and_store(self, file_path: str):
        documents = self.load_data(file_path)
        chunks = self.advanced_chunking(documents)
        embeddings = self.create_embeddings(chunks)
        self.setup_milvus()
        self.store_in_milvus(embeddings, chunks)
        print(f"Stored {len(chunks)} chunks in the Milvus database.")

    def get_all_docs(self):
        expr = "id >= 0"  
        res = self.collection.query(expr, output_fields=["id", "url", "title", "content", "embedding"])
        
        for doc in res:
            doc['embedding'] = [float(x) for x in doc['embedding']]  

        return res

    def clear_collection(self):
        logger.info("Clearing the collection")
        self.collection.delete(expr="id >= 0")
        self.collection.flush()
        file_path = "output.json"
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted.")
        else:
            print(f"File {file_path} does not exist.")
        # print(f"All entities in collection '{self.collection_name}' have been deleted.")

    def close_connection(self):
        connections.disconnect("default")

if __name__ == "__main__":
    handler = DataHandler()
    handler.process_and_store('output.json')
    handler.close_connection()