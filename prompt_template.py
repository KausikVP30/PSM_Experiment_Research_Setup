



def build_prompt(query, docs, past_answer=None):
    context = "\n\n".join(docs)

    if past_answer:
        prompt = f"""
You are a helpful AI assistant. Answer the question using the provided context.

Context:
{context}

Previous Answer:
{past_answer}

Question:
{query}

If the context contains the answer, use it. If not, say you don't have enough information.

Answer:
"""
    else:
        prompt = f"""
You are a helpful AI assistant. Answer the question using the provided context.

Context:
{context}

Question:
{query}

If the context contains the answer, use it. If not, say you don't have enough information.

Answer:
"""

    return prompt