from memory_retriever import MemoryRetriever
from hybrid_retriever import HybridRetriever
from memory_store import MemoryStore
from memory_index import MemoryIndex
from Embedding import EmbeddingModel
from prompt_template import build_prompt       # was missing
from llm_interface import LocalLLM            # was missing
import time
from datetime import datetime
from logs.logger import Logger


class Router:
    def __init__(self, documents, threshold=0.75):
        self.memory_retriever = MemoryRetriever(threshold=threshold)
        self.hybrid_retriever = HybridRetriever()
        self.hybrid_retriever.build_index(documents)   # was never called
        self.memory_store = MemoryStore()
        self.memory_index = MemoryIndex()
        self.embedding_model = EmbeddingModel()
        self.llm = LocalLLM()                          # was missing
        self.logger = Logger()
        self.retrieval_count = 0                     # was missing
        self.memory_count = 0                         # was missing

    def route(self, query):
        memory_result = self.memory_retriever.retrieve(query)

        if memory_result["use_memory"]:
            print("Using Memory...")
            return {
                "source": "memory",
                "docs": memory_result["retrieved_docs"],
                "answer": memory_result["answer"],
                "confidence": memory_result["confidence"]
            }
        else:
            print("Using Hybrid Retrieval...")
            docs = self.hybrid_retriever.retrieve(query)
            return {
                "source": "retrieval",
                "docs": docs,
                "answer": None,
                "confidence": memory_result["confidence"]
            }

    def store_memory(self, query, retrieved_docs, answer):
        query_embedding = self.embedding_model.encode_query(query)
        answer_embedding = self.embedding_model.encode_query(answer)

        self.memory_store.add_memory(
            query,
            query_embedding[0],
            retrieved_docs,
            answer,
            answer_embedding[0]
        )
        self.memory_index.add_memory_embedding(query_embedding[0])

    def process_query(self, query):
        start_time = time.time()

        memory_result = self.memory_retriever.retrieve(query)
        confidence = memory_result["confidence"]

        if memory_result["use_memory"]:
            print("Using Memory")
            docs = memory_result["retrieved_docs"]
            past_answer = memory_result["answer"]
            source = "memory"
            self.memory_count += 1
        else:
            print("Using Hybrid Retrieval")
            docs = self.hybrid_retriever.retrieve(query)
            past_answer = None
            source = "retrieval"                       # was missing
            self.retrieval_count += 1

        prompt = build_prompt(query, docs, past_answer)          # was missing
        answer = self.llm.generate(prompt)                       # was missing
        # Store memory if retrieval used
        if source == "retrieval":                              # was missing
            self.store_memory(query, docs, answer)

        end_time = time.time()
        latency = end_time - start_time

        # Log data
        self.logger.log(
            timestamp=str(datetime.now()),
            query=query,
            confidence=confidence,
            sim_q=memory_result.get("sim_query", 0),
            sim_a=memory_result.get("sim_answer", 0),
            sim_d=memory_result.get("sim_docs", 0),
            source=source,
            latency=latency,
            memory_size=self.memory_store.size(),              # was missing
            retrieval_count=self.retrieval_count                # was missing
        )

        return answer, confidence