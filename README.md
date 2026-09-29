# Electrical RAG — starter pipeline

A minimal retrieval-augmented pipeline for querying electrical documents
(GN3, PAT testing, BS 7671) grounded in the actual text, not the model's
memory.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install chromadb sentence-transformers pypdf anthropic
export ANTHROPIC_API_KEY=sk-ant-...   # get one from console.anthropic.com
```

## Usage

1. Ingest a document (run once per PDF):
   ```bash
   python ingest.py path/to/GN3.pdf --collection gn3
   ```

2. Ask questions against it:
   ```bash
   python ask.py "What tests are done before a ring final circuit is energised?" --collection gn3
   ```

3. Repeat step 1 for other documents (PAT testing, BS 7671 sections) with
   their own `--collection` name, e.g. `--collection pat`.

## What's deliberately left simple (for now)

- **One collection per document.** Querying across all of them at once,
  or letting the model decide which document is relevant, is the next
  step once single-document retrieval feels solid.
- **No decision-tree logic yet.** This is pure Q&A over text. The
  structured GN3 test-sequence flow (insulation resistance → continuity →
  polarity → RCD → Zs) is a separate layer to add once retrieval works
  well — that's more "if user is at step X, ask/check Y" logic than
  retrieval.
- **Local embeddings, cloud generation.** Chunking/embedding runs free on
  your machine; only the final answer generation calls the Claude API
  (a few cents per question). You could swap the embedding model out
  later for something more domain-tuned if generic embeddings
  under-perform on electrical terminology.
- **No web UI.** Command line only, so you can focus on whether the
  retrieval and answers are actually good before building an interface.

## Next steps once this works

- Try 10–15 real questions you'd expect a student to ask GN3 and see
  where retrieval misses (wrong page, missing chunk, question phrased
  differently than the text) — this tells you if chunk size/overlap
  needs tuning.
- Add the PAT testing document as a second collection.
- Start sketching the GN3 test-sequence as an explicit decision tree
  (a Python dict or state machine) that calls this Q&A pipeline for
  explanations at each step, rather than treating fault-finding as one
  big open-ended question.
