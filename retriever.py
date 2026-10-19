
import re

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

DB_PATH = "chroma_db"

embeddings = FastEmbedEmbeddings()

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)


def is_components_question(query):
    q = query.lower()

    return (
        ("component" in q or "components" in q)
        and (
            "rag" in q
            or "role does each" in q
            or "role of each" in q
            or "both" in q
        )
    )


def search_rag_paper(search_queries, k=5):
    """Retrieve and deduplicate chunks from the RAG paper."""
    results = []

    for search_query in search_queries:
        docs = vectorstore.similarity_search(search_query, k=k)

        print(f"\nSEARCH QUERY: {search_query}")

        for doc in docs:
            source = doc.metadata.get("source", "").replace("\\", "/")

            print("SOURCE:", source)
            print("PAGE:", doc.metadata.get("page"))
            print("CHUNK:", doc.page_content[:300])

            if "2005.11401v4.pdf" in source:
                results.append(doc)

    unique_docs = []
    seen = set()

    for doc in results:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content
        )

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def extract_section(doc, section_number, section_name, next_section_number,
                    required_terms):
    """Extract a specific numbered section when its heading is available."""
    content = doc.page_content

    heading_pattern = (
        rf"{re.escape(section_number)}\s*"
        rf"(?:\r?\n\s*)?"
        rf"{re.escape(section_name)}"
    )

    match = re.search(
        heading_pattern,
        content,
        flags=re.IGNORECASE
    )

    if match:
        section_text = content[match.start():]

        next_heading = re.search(
            rf"(?:^|\n)\s*{re.escape(next_section_number)}"
            rf"\s*(?:\r?\n|[A-Z])",
            section_text[match.end() - match.start():],
            flags=re.IGNORECASE
        )

        if next_heading:
            end = match.end() - match.start() + next_heading.start()
            section_text = section_text[:end]

        content_to_check = section_text.lower()

        if all(term in content_to_check for term in required_terms):
            return Document(
                page_content=section_text.strip(),
                metadata=doc.metadata
            )

    # If the heading isn't present in this chunk, accept it only when
    # it contains the distinctive content required for this section.
    lower = content.lower()

    if all(term in lower for term in required_terms):
        return doc

    return None


def retrieve_components():
    """Retrieve distinct evidence for both RAG components."""
    retriever_queries = [
        "DPR follows a bi-encoder architecture",
        "Retriever: DPR document encoder query encoder",
        "retrieval component retrieves text passages given a query"
    ]

    generator_queries = [
        "The generator component uses BART-large",
        "Generator: BART encoder-decoder architecture",
        "BART-large generates target sequence using retrieved documents"
    ]

    retriever_chunks = search_rag_paper(retriever_queries, k=5)
    generator_chunks = search_rag_paper(generator_queries, k=5)

    retriever_docs = []
    for doc in retriever_chunks:
        section_doc = extract_section(
            doc,
            "2.2",
            "Retriever: DPR",
            "2.3",
            ["bi-encoder architecture"]
        )

        if section_doc:
            retriever_docs.append(section_doc)

    generator_docs = []
    for doc in generator_chunks:
        section_doc = extract_section(
            doc,
            "2.3",
            "Generator: BART",
            "2.4",
            ["bart-large"]
        )

        if section_doc:
            generator_docs.append(section_doc)

    def deduplicate(docs):
        unique = []
        seen = set()

        for doc in docs:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content
            )

            if key not in seen:
                seen.add(key)
                unique.append(doc)

        return unique

    retriever_docs = deduplicate(retriever_docs)
    generator_docs = deduplicate(generator_docs)

    print("\nRETRIEVER SECTION MATCHES:", len(retriever_docs))
    print("GENERATOR SECTION MATCHES:", len(generator_docs))

    for doc in retriever_docs:
        print("\nRETRIEVER EVIDENCE:", doc.page_content[:700])

    for doc in generator_docs:
        print("\nGENERATOR EVIDENCE:", doc.page_content[:700])

    # Return both sections together only if both were found.
    if retriever_docs and generator_docs:
        return retriever_docs[:2] + generator_docs[:2]

    # Never silently substitute an unrelated introduction or table.
    print("Could not retrieve both component sections.")
    return []


def retrieve(query):
    query_lower = query.lower()

    # Handle combined questions before individual component questions.
    if is_components_question(query):
        return retrieve_components()

    generator_terms = [
        "generator",
        "bart",
        "bart-large",
        "generation",
        "rag-sequence",
        "rag-token"
    ]

    retriever_terms = [
        "retriever",
        "dpr",
        "bi-encoder",
        "retrieval architecture"
    ]

    is_generator_question = any(
        term in query_lower for term in generator_terms
    )

    is_retriever_question = any(
        term in query_lower for term in retriever_terms
    )

    if is_generator_question or is_retriever_question:
        if is_generator_question:
            search_queries = [
                "Generator: BART",
                "BART-large pre-trained seq2seq transformer",
                "The generator component uses BART-large"
            ]
        else:
            search_queries = [
                "Retriever: DPR bi-encoder architecture",
                "The retrieval component is based on DPR",
                "DPR document encoder query encoder BERTBASE"
            ]

        unique_docs = search_rag_paper(search_queries, k=5)

        if is_generator_question:
            generator_docs = []

            for doc in unique_docs:
                section_doc = extract_section(
                    doc,
                    "2.3",
                    "Generator: BART",
                    "2.4",
                    ["bart-large"]
                )

                if section_doc:
                    generator_docs.append(section_doc)

            if generator_docs:
                return generator_docs[:4]

        if is_retriever_question:
            dpr_docs = [
                doc for doc in unique_docs
                if (
                    "bi-encoder architecture" in doc.page_content.lower()
                    and (
                        "retriever: dpr" in doc.page_content.lower()
                        or "retrieval component" in doc.page_content.lower()
                        or "dpr follows" in doc.page_content.lower()
                    )
                )
            ]

            if dpr_docs:
                return dpr_docs[:4]

        if unique_docs:
            return unique_docs[:8]

    # Standard retrieval for other questions.
    return vectorstore.similarity_search(query, k=8)
