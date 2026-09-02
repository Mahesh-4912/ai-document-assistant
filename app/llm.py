from ollama import chat


def generate_answer(question, context):
    prompt = f"""
You are an AI Document Assistant.

Answer the user's question using only the provided document context.

Keep the answer simple, clear, and easy to understand.

If the answer is not available in the context, say:
"I could not find this information in the document."

Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = chat(
        model="qwen3:0.6b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content