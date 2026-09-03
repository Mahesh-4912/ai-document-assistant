from ollama import chat


def choose_action(question):
    prompt = f"""
You are an AI document assistant agent.

Choose exactly one action for the user's request.

Available actions:

ASK
Use when the user asks a question about information
inside the document.

SUMMARIZE
Use when the user asks to summarize the document.

Return only one word:
ASK
or
SUMMARIZE

User request:
{question}
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

    action = response.message.content.strip().upper()

    if "SUMMARIZE" in action:
        return "SUMMARIZE"

    return "ASK"