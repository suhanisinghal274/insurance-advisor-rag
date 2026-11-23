"""The advisor assistant: retrieve policy sections, answer with citations, or hand off to a human."""
import os
import time

from src import config
from src.ingest import build_collection

SYSTEM_PROMPT = """You are an assistant for insurance advisors at Lumina Assurance.
Answer ONLY from the policy excerpts provided. Be concise and exact with numbers.
After each fact, cite its source in square brackets as [Policy name, section].
If the excerpts do not contain the answer, reply with exactly one word: ESCALATE
Never guess, and never give personal financial advice."""

ESCALATION_MESSAGE = "I can't answer this reliably from the policy documents. Flagged for a human advisor to review."


class GeminiLLM:
    def __init__(self):
        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def __call__(self, prompt):
        from google.genai import errors
        # Retry on rate limits (429) and temporary overload (500/503), waiting 5s, 10s, 20s, 40s
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=self.genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0),
                )
                return (resp.text or "").strip()
            except errors.APIError as e:
                if e.code in (429, 500, 503) and attempt < config.MAX_RETRIES:
                    wait = 5 * 2 ** attempt
                    print(f"Gemini returned {e.code}; retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise


class MockLLM:
    """Offline stand-in: returns the top excerpt. Used only to test the pipeline without an API key."""

    def __call__(self, prompt):
        first = prompt.split("[1]", 1)[1].split("\n[2]", 1)[0]
        header, _, body = first.partition("\n")
        return f"{body.strip()} [{header.strip(' ()')}]"


class AdvisorAssistant:
    def __init__(self, persist=True):
        self.collection, self.chunks = build_collection(persist=persist)
        self.llm = MockLLM() if config.LLM_BACKEND == "mock" else GeminiLLM()

    def retrieve(self, question, k=config.TOP_K):
        res = self.collection.query(query_texts=[question], n_results=k)
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"id": cid, "text": doc, "score": round(1 - dist, 4), **meta})
        return hits

    def answer(self, question):
        start = time.perf_counter()
        hits = self.retrieve(question)
        top_score = hits[0]["score"] if hits else 0.0

        if top_score < config.ESCALATE_BELOW:
            escalated, reason, text = True, "low retrieval confidence", ESCALATION_MESSAGE
        else:
            context = "\n".join(
                f"[{i}] ({h['policy']}, {h['section']})\n{h['text']}" for i, h in enumerate(hits, 1)
            )
            prompt = f"Policy excerpts:\n{context}\n\nAdvisor question: {question}"
            text = self.llm(prompt)
            escalated = text.strip().upper().startswith("ESCALATE")
            reason = "model found no answer in the excerpts" if escalated else None
            if escalated:
                text = ESCALATION_MESSAGE

        return {
            "question": question,
            "answer": text,
            "escalated": escalated,
            "escalation_reason": reason,
            "top_score": top_score,
            "sources": [{"id": h["id"], "policy": h["policy"], "section": h["section"], "score": h["score"]} for h in hits],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }


if __name__ == "__main__":
    bot = AdvisorAssistant()
    q = "What is the waiting period for cancer under the critical illness plan?"
    print(bot.answer(q))
