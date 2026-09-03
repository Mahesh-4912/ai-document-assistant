from ollama import chat


MODEL_NAME = "qwen3:0.6b"


def generate_answer(question, context):
    prompt = f"""
You are an AI Document Assistant.

Answer the question using ONLY the document context.

Important rules:
- Do not invent information.
- Do not confuse number of projects with years of experience.
- If exact information is not available, say:
"I could not find this information in the document."
- Keep the answer clear and concise.

DOCUMENT CONTEXT:
----------------
{context}
----------------

USER QUESTION:
{question}

ANSWER:
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


def summarize_document(context):
    prompt = f"""
You are an AI Document Assistant.

Summarize the following document.

Rules:
- Use only information from the document.
- Do not invent facts.
- Mention important skills, experience,
  education or key topics when available.
- Keep the summary concise and structured.

DOCUMENT:
----------------
{context}
----------------

SUMMARY:
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content