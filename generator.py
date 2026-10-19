
import re
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

INSUFFICIENT_EVIDENCE = (
    "The retrieved context does not contain sufficient evidence to answer."
)

SYSTEM_PROMPT = """
You are a PDF Knowledge Assistant.
Answer concisely using only the retrieved context.
Do not invent facts or add unsupported claims.
If the context is insufficient, say:
"The retrieved context does not contain sufficient evidence to answer."
Return only the explanation.
"""


def normalize(text):
    return " ".join((text or "").split())


def is_process_question(query):
    q = query.lower()
    return "rag" in q and any(term in q for term in [
        "how", "retrieve", "retrieval", "generate", "generation"
    ])


def is_components_question(query):
    q = query.lower()
    return "rag" in q and any(term in q for term in [
        "two main components", "both components",
        "each component", "role of each", "role does each"
    ])


def split_sentences(text):
    return [
        normalize(s)
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if normalize(s)
    ]


def find_overview(docs):
    for doc in docs:
        content = doc.page_content
        lower = content.lower()

        start = lower.find("we explore rag models")
        if start == -1:
            continue

        end_candidates = [
            lower.find("1code to run experiments", start),
            lower.find("2.1 ", start),
        ]
        end_candidates = [p for p in end_candidates if p > start]
        end = min(end_candidates) if end_candidates else len(content)

        excerpt = normalize(content[start:end])

        if (
            "retrieve text documents" in excerpt.lower()
            and "additional context" in excerpt.lower()
            and "generator" in excerpt.lower()
        ):
            return excerpt, doc

    return None, None


def find_generation_process(docs):
    for doc in docs:
        content = doc.page_content
        lower = content.lower()

        start = lower.find(
            "concretely, the top k documents are retrieved"
        )
        if start == -1:
            start = lower.find("top k documents are retrieved")
        if start == -1:
            continue

        excerpt = content[start:]
        excerpt_lower = excerpt.lower()

        end_candidates = [
            excerpt_lower.find("finally, we note"),
            excerpt_lower.find("2.2 retriever"),
            excerpt_lower.find("2.3 generator"),
        ]
        end_candidates = [p for p in end_candidates if p > 0]
        if end_candidates:
            excerpt = excerpt[:min(end_candidates)]

        excerpt = normalize(excerpt)

        if (
            "generator produces a distribution" in excerpt.lower()
            and "top k documents" in excerpt.lower()
        ):
            return excerpt, doc

    return None, None


def extract_evidence(query, docs):
    q = query.lower()

    if is_process_question(query) or is_components_question(query):
        overview, overview_doc = find_overview(docs)
        process, process_doc = find_generation_process(docs)

        if is_components_question(query) and overview:
            return overview, [overview_doc]

        if not overview or not process:
            return None, None

        evidence = (
            f"RAG overview:\n{overview}\n\n"
            f"Generation process:\n{process}"
        )

        evidence_docs = [overview_doc]
        if process_doc is not overview_doc:
            evidence_docs.append(process_doc)

        return evidence, evidence_docs

    if any(term in q for term in [
        "generator", "bart", "generation", "rag-token", "rag-sequence"
    ]):
        for doc in docs:
            sentences = split_sentences(doc.page_content)
            matches = [
                s for s in sentences
                if (
                    ("bart-large" in s.lower()
                     or "generator component" in s.lower())
                    and len(s.split()) >= 8
                )
            ]
            if matches:
                return " ".join(matches[:2]), [doc]
        return None, None

    if any(term in q for term in [
        "retriever", "dpr", "bi-encoder", "retrieval architecture"
    ]):
        for doc in docs:
            match = re.search(
                r"DPR follows a bi-encoder architecture"
                r"[^.!?]*[.!?]",
                doc.page_content,
                flags=re.IGNORECASE
            )
            if match and len(match.group(0).split()) >= 8:
                return normalize(match.group(0)), [doc]
        return None, None

    # Conservative fallback for other questions.
    terms = {
        w for w in re.findall(r"[a-z0-9]+", q)
        if len(w) > 3 and w not in {
            "what", "which", "when", "where", "does", "used",
            "from", "that", "this", "with", "about"
        }
    }

    for doc in docs:
        for sentence in split_sentences(doc.page_content):
            overlap = terms & set(
                re.findall(r"[a-z0-9]+", sentence.lower())
            )
            if len(sentence.split()) >= 8 and len(overlap) >= 2:
                return sentence, [doc]

    return None, None


def clean_explanation(text):
    text = (text or "").strip()
    text = re.sub(
        r"^\s*Explanation\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.split(
        r"\bExact evidence\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]
    return text.strip()


def identify_section(doc):
    lower = doc.page_content.lower()

    if "we explore rag models" in lower:
        return "2 — Methods"
    if "retriever: dpr" in lower or "bi-encoder architecture" in lower:
        return "2.2 — Retriever: DPR"
    if "generator: bart" in lower or "bart-large" in lower:
        return "2.3 — Generator: BART"

    return "Not identified in retrieved text"


def generate_answer(query, docs):
    if not docs:
        return INSUFFICIENT_EVIDENCE

    evidence, evidence_docs = extract_evidence(query, docs)

    if not evidence or not evidence_docs:
        return (
            f"Explanation:\n{INSUFFICIENT_EVIDENCE}\n\n"
            "Exact evidence:\n"
            "No relevant supporting evidence was found in the retrieved text."
        )

    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'Unknown')}, "
        f"Page: {d.metadata.get('page', 'Unknown')}]\n"
        f"{d.page_content}"
        for d in docs
    )

    try:
        response = client.chat.completions.create(
            model="phi-3.1-mini-4k-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{query}\n\n"
                        "Answer in 2 to 4 sentences. Explain only facts "
                        "supported by the context. Do not add claims about "
                        "accuracy or benefits unless explicitly supported."
                    )
                }
            ],
            temperature=0,
            max_tokens=160
        )

        explanation = clean_explanation(
            response.choices[0].message.content
        )

        metadata = []
        for doc in evidence_docs:
            metadata.append(
                f"Source: {doc.metadata.get('source', 'Unknown source')}\n"
                f"Section: {identify_section(doc)}\n"
                f"Page metadata: {doc.metadata.get('page', 'Unknown page')}"
            )

        metadata = list(dict.fromkeys(metadata))

        return (
            f"Explanation:\n{explanation}\n\n"
            f"Exact evidence:\n“{evidence}”\n\n"
            + "\n\n".join(metadata)
        )

    except Exception as e:
        return f"LLM error: {e}"
