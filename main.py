from router import Router

DOCUMENTS = [
    "A linked list is a linear data structure where elements are stored in nodes.",
    "Each node in a linked list contains data and a pointer to the next node.",
    "A doubly linked list has pointers to both the next and previous nodes.",
    "Binary search trees allow fast lookup, insertion, and deletion of elements.",
    "Stacks follow Last In First Out (LIFO) principle for data storage.",
    "Queues follow First In First Out (FIFO) principle for data storage.",
    "Hash tables use a hash function to map keys to values for fast retrieval.",
    "Graph traversal algorithms include BFS and DFS for visiting all nodes.",
    "Dynamic programming breaks problems into overlapping subproblems.",
    "Bubble sort repeatedly swaps adjacent elements if they are in wrong order.",
]

if __name__ == "__main__":
    router = Router(documents=DOCUMENTS)

    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        answer, confidence = router.process_query(query)

        print(f"\nConfidence : {confidence:.2f}")
        print(f"Answer     : {answer}")