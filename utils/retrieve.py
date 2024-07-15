import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, DPRQuestionEncoder
from rank_bm25 import BM25Okapi
import torch
import nltk
from nltk.corpus import wordnet
from utils.data import DataHandler
import numpy as np
import warnings
from utils.logger import logger

# Set environment variable to control tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Retrieval:
    """
    A sophisticated information retrieval system for NVIDIA CUDA documentation.

    This class implements a hybrid retrieval approach, combining BM25 and Dense 
    Passage Retrieval (DPR) methods, followed by a re-ranking step using a 
    cross-encoder model.

    Attributes:
        data_handler (DataHandler): Handles interactions with the vector database.
        tokenizer (AutoTokenizer): Tokenizer for the DPR model.
        question_encoder (DPRQuestionEncoder): Encodes questions for DPR.
        cross_encoder (SentenceTransformer): Used for re-ranking results.

    """
    def __init__(self, collection_name: str = "nvidia_cuda_docs"):
        logger.info("Intialized Retrieve class")
        self.data_handler = DataHandler(collection_name)
        
        # Initialize models
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        
        self.cross_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize NLTK
        nltk.download('wordnet', quiet=True)

        # Print collection info
        # self.print_collection_info()

    def print_collection_info(self):
        num_entities = self.data_handler.collection.num_entities
        schema = self.data_handler.collection.schema
        print(f"Number of entities in collection: {num_entities}")
        print("Schema of collection:")
        for field in schema.fields:
            print(f"- {field.name}: {field.dtype} (primary: {field.is_primary})")

    def query_expansion(self, query: str) -> List[str]:
        """
        Expand the given query using WordNet synonyms.

        Args:
            query (str): The original query string.

        Returns:
            List[str]: A list of expanded queries including the original.
        """
        expanded_queries = [query]
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_queries.append(query.replace(word, lemma.name()))
        return list(set(expanded_queries))

    def bm25_retrieval(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        all_docs = self.data_handler.get_all_docs()
        
        documents = [doc['content'] for doc in all_docs]
        doc_ids = [doc['id'] for doc in all_docs]
        
        tokenized_corpus = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        doc_scores = bm25.get_scores(tokenized_query)
        top_n = sorted(zip(doc_ids, doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
        return top_n

    def dpr_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        inputs = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            question_encoder_output = self.question_encoder(**inputs)
            question_embedding = question_encoder_output.pooler_output
            question_embedding = question_embedding.detach().cpu().numpy().tolist()[0]
        
        search_params = {"metric_type": "L2", "params": {"ef": 100}}
        results = self.data_handler.collection.search(
            data=[question_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["url", "title"]
        )
        
        return results[0]

    def hybrid_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        """
        Perform hybrid retrieval combining BM25 and DPR results.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict]: Combined and sorted list of search results.
        """

        logger.info("Retrieving results from BM25 and DPR")
        bm25_results = self.bm25_retrieval(query, top_k=top_k)
        dpr_results = self.dpr_retrieval(query, top_k=top_k)
        
        combined_results = {}
        max_bm25_score = max((score for _, score in bm25_results), default=1)
        max_dpr_score = max((hit.distance for hit in dpr_results), default=1)
        
        for i, score in bm25_results:
            combined_results[i] = score / max_bm25_score if max_bm25_score != 0 else 0
        
        for hit in dpr_results:
            doc_id = hit.id
            if doc_id in combined_results:
                combined_results[doc_id] += (1 - hit.distance / max_dpr_score) if max_dpr_score != 0 else 0
            else:
                combined_results[doc_id] = (1 - hit.distance / max_dpr_score) if max_dpr_score != 0 else 0
        
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": doc_id, "score": score} for doc_id, score in sorted_results]

    def rerank(self, query: str, results: List[Dict], documents: Dict[int, str]) -> List[Dict]:
        """
        Re-rank the retrieved results using a cross-encoder model.

        Args:
            query (str): The original query string.
            results (List[Dict]): The initial retrieval results.
            documents (Dict[int, str]): Dictionary of document contents.

        Returns:
            List[Dict]: Re-ranked list of search results.
        """
        logger.info("Reranking the results")
        pairs = [(query, documents[result['id']]) for result in results if result['id'] in documents]
        
        query_embeddings = self.cross_encoder.encode([pair[0] for pair in pairs])
        doc_embeddings = self.cross_encoder.encode([pair[1] for pair in pairs])
        
        similarity_scores = util.pytorch_cos_sim(query_embeddings, doc_embeddings)
        
        for i, score in enumerate(similarity_scores[0]):
            results[i]['rerank_score'] = float(score)
        
        reranked_results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked_results

    def retrieve_and_rerank(self, query: str, top_k: int = 100) -> List[Dict]:
        """
        Perform the complete retrieval and re-ranking process.

        This method combines query expansion, hybrid retrieval, and re-ranking
        to produce the final set of relevant documents.

        Args:
            query (str): The original query string.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict]: Final list of ranked and relevant documents.
        """
        expanded_queries = self.query_expansion(query)
        
        all_results = []
        for exp_query in expanded_queries:
            results = self.hybrid_retrieval(exp_query, top_k=top_k)
            all_results.extend(results)
        
        unique_results = list({r['id']: r for r in all_results}.values())
        top_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # Get only the documents for the top results
        doc_ids = [result['id'] for result in top_results]
        docs = self.data_handler.collection.query(
            expr=f"id in {doc_ids}",
            output_fields=["*"]
        )
        documents = {doc['id']: doc['content'] for doc in docs}
        
        final_results = self.rerank(query, top_results, documents)
        logger.info(f"Retrieved and reranked {len(final_results)} results")
        
        return final_results

if __name__ == "__main__":
    retriever = Retrieval()
    query = "How to optimize CUDA kernel performance?"
    results = retriever.retrieve_and_rerank(query)
    for i, result in enumerate(results[:10], 1):
        print(f"{i}. Document ID: {result['id']}, Score: {result['rerank_score']:.4f}")