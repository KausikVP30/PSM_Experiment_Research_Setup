from memory_index import MemoryIndex
from memory_store import MemoryStore

class MemoryRetriever:
    def __init__(self, threshold=0.75):
        self.memory_index = MemoryIndex()
        self.memory_store = MemoryStore()
        self.threshold = threshold

    def retrieve(self, query):
        memory_id, similarity = self.memory_index.search_memory(query)

        # If no memory exists
        if memory_id is None:
            return {
                "use_memory": False,
                "confidence": 0,
                "retrieved_docs": None,
                "answer": None,
                "memory_id": None
            }

        # If confidence high → use memory
        if similarity >= self.threshold:
            memory_entry = self.memory_store.get_memory_by_id(memory_id)

            return {
                "use_memory": True,
                "confidence": similarity,
                "retrieved_docs": memory_entry["retrieved_docs"],
                "answer": memory_entry["answer"],
                "memory_id": memory_id
            }

        # If confidence low → do retrieval
        else:
            return {
                "use_memory": False,
                "confidence": similarity,
                "retrieved_docs": None,
                "answer": None,
                "memory_id": memory_id
            }