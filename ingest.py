"""
ingest.py — Step 1 of the pipeline: turn a PDF into searchable chunks.

Usage:
    python ingest.py path/to/GN3.pdf --collection gn3

What this does:
    1. Extracts text from the PDF, page by page.
    2. Splits it into overlapping chunks (so context isn't cut mid-sentence).
    3. Embeds each chunk into a vector (a list of numbers capturing meaning).
    4. Stores the chunks + vectors in a local Chroma database on disk.

Run this once per document. Re-run with --collection <name> for each new
source (gn3, pat, bs7671) so you can query them separately or together later.
"""

import argparse
import pathlib
import sys

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from pypdf.errors import PdfReadError

DB_PATH = "./chroma_db"  # local folder — all your data stays on your machine
CHUNK_SIZE = 800          # characters per chunk — small enough for precise retrieval
CHUNK_OVERLAP = 150       # overlap so a sentence split across chunks isn't lost


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) tuples.

    Raises SystemExit with a clear message on encrypted/corrupted PDFs,
    instead of letting a raw traceback reach the user.
    """
    try:
        reader = PdfReader(pdf_path)
    except PdfReadError as e:
        raise SystemExit(f"Could not open '{pdf_path}' — it may be corrupted: {e}")

    if reader.is_encrypted:
        raise SystemExit(
            f"'{pdf_path}' is password-protected. Remove the password "
            "(e.g. with 'qpdf --decrypt') before ingesting it."
        )

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the vector store")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--collection",
        required=True,
        help="Name for this document's collection, e.g. gn3, pat, bs7671",
    )
    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf_path)
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    print(f"Reading {pdf_path.name}...")
    pages = extract_pages(str(pdf_path)) 
    print(f"  {len(pages)} pages with extractable text")

    # Build chunks, keeping track of which page each chunk came from.
    all_chunks = []
    all_metadata = []
    for page_num, text in pages:
        for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            if chunk.strip():
                all_chunks.append(chunk)
                all_metadata.append({"source": pdf_path.name, "page": page_num})

        print(f"  {len(all_chunks)} chunks created")

    # Fail loudly and clearly if the PDF produced nothing usable, rather than
    # calling upsert() with empty lists (which errors confusingly).
    if not all_chunks:
        raise SystemExit(
            f"No text could be extracted from '{pdf_path.name}'. "
            "It may be a scanned/image-only PDF — run OCR on it first."
        )

    # Local, free embedding model — runs on your machine, no API key needed.
    # (First run downloads the model, ~80MB, then it's cached.)
    try:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception as e:
        raise SystemExit(
            "Could not load the embedding model. If this is the first run, "
            f"it needs an internet connection to download it. Error: {e}"
        )

    client = chromadb.PersistentClient(path=DB_PATH)

    # Re-ingesting the same collection name drops the old version first.
    # Without this, upsert() only overwrites IDs that still exist — if the
    # new document produces fewer chunks than before, leftover chunks from
    # the previous run would silently stay in the database forever.
    existing = [c.name for c in client.list_collections()]
    if args.collection in existing:
        print(f"Collection '{args.collection}' already exists — replacing it.")
        client.delete_collection(name=args.collection)

    collection = client.get_or_create_collection(
        name=args.collection, embedding_function=embed_fn
    )

    ids = [f"{args.collection}-{i}" for i in range(len(all_chunks))]

    print("Embedding and storing chunks (may take a minute)...")
    # Upsert in batches so very large documents (thousands of pages) don't
    # risk hitting Chroma's per-call batch-size limit or using excess memory
    # all at once.
    BATCH_SIZE = 500
    for start in range(0, len(all_chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            documents=all_chunks[start:end],
            metadatas=all_metadata[start:end],
            ids=ids[start:end],
        )
        print(f"  stored {min(end, len(all_chunks))}/{len(all_chunks)} chunks")

    print(f"Done. Collection '{args.collection}' now has {collection.count()} chunks.")
    print(f"Stored at: {DB_PATH}")

if __name__ == "__main__":
    main() 