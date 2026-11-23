# Insurance Advisor Assistant (RAG with human hand-off)

An assistant for insurance advisors that answers product questions from policy documents, cites the exact section it used, and **hands the question to a human advisor when it isn't confident** instead of guessing. It comes with an evaluation script that measures retrieval quality, answer accuracy, hand-off precision and recall, task completion and latency.

> All policies belong to **Lumina Assurance, a fictional insurer**. The six policy documents in `data/policies/` are synthetic sample data written for this project.

Built with Claude (Anthropic) as an AI pair-programmer.

## How it works

1. **Ingest** (`src/ingest.py`): each policy is split into its numbered sections, embedded with `all-MiniLM-L6-v2` (sentence-transformers) and stored in a **ChromaDB** vector database.
2. **Retrieve** (`src/rag.py`): the advisor's question is embedded, and the top 4 most similar sections are retrieved by cosine similarity.
3. **Decide**: if the best match scores below a threshold, the question is handed to a human. Otherwise the sections go to **Gemini** with instructions to answer only from them, cite every fact, and reply `ESCALATE` if the answer isn't there.
4. **Evaluate** (`src/evaluate.py`): 30 test questions (24 answerable, 6 the documents can't answer) are scored automatically.

**Reliability:** model calls retry temporary overload (503) and per-minute limits (429) with exponential backoff; a daily-quota error stops the run cleanly, and saved answers let the evaluation resume without re-sending requests. Retry waits are excluded from latency.

## Evaluation

Run `python -m src.evaluate` (or section 6 of the Colab notebook). Results are written to `eval/results/`.

| Metric | What it measures | Result |
|---|---|---|
| Retrieval recall@4 | Expected section is in the top 4 retrieved | _fill in from your run_ |
| MRR | How high the expected section ranks | _fill in_ |
| Answer accuracy | Answerable questions answered with every expected fact | _fill in_ |
| Hand-off precision | Of all hand-offs, how many were truly unanswerable | _fill in_ |
| Hand-off recall | Of unanswerable questions, how many were handed off | _fill in_ |
| Task completion | Correct answer or correct hand-off, over all 30 | _fill in_ |
| Latency p50 / p95 | Time per question | _fill in_ |

## Run it

**Google Colab (easiest):** open `run_in_colab.ipynb` in Colab, add a free Gemini API key (from Google AI Studio) to Colab Secrets as `GEMINI_API_KEY`, and run the cells in order.

**Locally:**
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key      # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
python -m src.ingest
python -m src.evaluate
python app.py                        # web demo on http://localhost:7860
```

**Docker:**
```bash
docker build -t insurance-advisor-rag .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key insurance-advisor-rag
```

**Offline test (no API key, no model download):** `EMBEDDING_BACKEND=tfidf LLM_BACKEND=mock python -m src.evaluate`

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.5-flash` | Answer model |
| `TOP_K` | `4` | Sections retrieved per question |
| `ESCALATE_BELOW` | `0.35` | Hand off if the best section scores below this |
| `EMBEDDING_BACKEND` | `minilm` | `minilm` or `tfidf` (offline) |
| `LLM_BACKEND` | `gemini` | `gemini`, `groq` (needs `GROQ_API_KEY`) or `mock` (offline) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Answer model when using Groq |
| `MAX_RETRIES` | `4` | Retries on temporary overload or per-minute limits |
| `EVAL_DELAY_S` | `4` | Pause between evaluation questions |

## Limitations and next steps

- Answer accuracy uses keyword matching, which is strict and simple. A next step is an LLM-as-judge score checked against human labels.
- Six short documents only; real policy wordings are longer and need better chunking.
- Next: multi-turn conversation, a per-product filter, and logging live questions to monitor drift.
