"""Prompt text and prompt-building helpers."""

NOT_FOUND = "I could not find this information in the provided learning materials."

SYSTEM_PROMPT = f"""You are an AI and Machine Learning Learning Assistant.

Your target users are university students studying Artificial Intelligence,
Machine Learning, Data Analytics, and related subjects.

Your rules:
1. Explain concepts in simple and clear language.
2. Give step-by-step explanations where useful.
3. Use only the retrieved knowledge base context when answering.
4. If the answer is not available in the retrieved context, say exactly:
   "{NOT_FOUND}"
5. Do not make up facts.
6. Encourage learning, not copying.
7. Be polite, supportive, and student-friendly.
8. If the question is unclear, ask the learner to clarify.
"""


def format_context(sources) -> str:
    """Turn retrieved Source objects into one text block for the prompt."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(f"[Document {i} | {s.file} | {s.location}]\n{s.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> list[dict]:
    user_prompt = f"""Retrieved learning material:
{context}

Learner question:
{question}

Answer the question using ONLY the retrieved learning material above.
If the material does not contain enough information to answer, reply EXACTLY with:
{NOT_FOUND}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
