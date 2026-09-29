"""
ask.py — Step 2 of the pipeline: ask a question, get an answer grounded in
your ingested documents.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

What this does:
    1. Embeds your question the same way the chunks were embedded.
    2. Finds the most similar chunks in the vector store (retrieval).
    3. Passes those chunks + your question to Claude, instructed to answer
       ONLY from the provided material and cite the page number.
"""

import argparse
import os

import anthropic
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./chroma_db"
TOP_K = 5  # how many chunks to retrieve — start small, can be tuned.

SYSTEM_PROMPT = """You are a study aid for an electrician answering questions \
about UK electrical testing and regulations. You will be given excerpts from \
a source document and a question.

Rules:
- Answer ONLY using the provided excerpts. If they don't contain the answer,
say so plainly rather than guessing or using outside knowledge.
- Always cite the page number(s) your answer comes from.
- This is a study aid, not a substitute for a competent person or safe
isolation procedure. If the question involves live testing or safety-critical
steps, remind the user to follow proper safe isolation procedure.
- Be concise and precise — this is for someone training in the trade, not a
general audience.
"""


def main():
    parser = argparse.ArgumentParser(description="Ask a question against a collection")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--collection", required=True, help="Which collection to search")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full retrieved chunks sent to Claude, not just page numbers",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment first.")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        collection = client.get_collection(name=args.collection, embedding_function=embed_fn)
    except Exception:
        raise SystemExit(
            f"Collection '{args.collection}' not found. Run ingest.py first."
        )

    results = collection.query(query_texts=[args.question], n_results=args.top_k, include=["documents", "metadatas", "distances"],)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not chunks:
        raise SystemExit("No relevant chunks found.")

    # Chroma always returns its top-K *nearest* chunks, even if none of them
    # are actually a good match — it has no concept of "not relevant enough".
    # A rough distance check catches the case where every result is a weak
    # match, so you get a warning instead of a confident answer built on irrelevant text. 
    # This threshold is an approximation for this embedding
    # model (all-MiniLM-L6-v2) — treat it as a starting point to tune once
    # you've seen how distances look for questions you know are answerable.
    WEAK_MATCH_THRESHOLD = 1.3
    best_distance = min(distances)
    if best_distance > WEAK_MATCH_THRESHOLD:
        print(
            "⚠️  Warning: even the closest match is a weak one "
            f"(distance {best_distance:.2f}). This document may not actually "
            "cover this question — treat the answer below with extra caution.\n"
        )

    # Build the context block Claude will see, with page numbers attached.
    context_parts = []
    for chunk, meta in zip(chunks, metadatas):
        context_parts.append(f"[Page {meta['page']}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    print(
        f"Retrieved {len(chunks)} chunks "
        f"(pages: {[m['page'] for m in metadatas]}, "
        f"distances: {[round(d, 2) for d in distances]})\n"
    )

    if args.verbose:
        print("--- Full retrieved context sent to Claude ---")
        print(context)
        print("--- End of context ---\n")

    try:
        ai_client = anthropic.Anthropic()
        response = ai_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Excerpts:\n\n{context}\n\nQuestion: {args.question}",
                }
            ],
        )
    except anthropic.APIError as e:
        raise SystemExit(f"Claude API call failed: {e}")

    answer = "".join(block.text for block in response.content if block.type == "text")
    print("Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
