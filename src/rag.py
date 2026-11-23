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


class QuotaExhausted(Exception):
    """The model's daily free-tier quota is used up; retrying in seconds won't help."""


def _with_retries(call, name):
    """Run call(); retry temporary overload and per-minute limits with waits of 5s, 10s, 20s, 40s.
    Returns (text, seconds spent waiting) so waiting isn't counted as latency."""
    waited = 0.0
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            return call(), waited
        except Exception as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            msg = str(e).lower()
            if code == 429 and ("per day" in msg or "perday" in msg):
                raise QuotaExhausted(f"{name}: daily free-tier quota used up. {str(e)[:200]}") from e
            if code in (429, 500, 502, 503) and attempt < config.MAX_RETRIES:
                wait = 5 * 2 ** attempt
                print(f"{name} returned {code}; retrying in {wait}s...")
                time.sleep(wait)
                waited += wait
                continue
            raise


class GeminiLLM:
    model = config.GEMINI_MODEL

    def __init__(self):
        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def __call__(self, prompt):
        def call():
            resp = self.client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=self.genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0),
            )
            return (resp.text or "").strip()
        return _with_retries(call, "Gemini")


class GroqError(Exception):
    def __init__(self, status_code, text):
        super().__init__(f"Groq {status_code}: {text[:300]}")
        self.status_code = status_code


class GroqLLM:
    """Free alternative (console.groq.com) through its OpenAI-compatible endpoint; needs GROQ_API_KEY."""
    model = config.GROQ_MODEL

    def __init__(self):
        import requests
        self.requests = requests
        self.key = os.environ["GROQ_API_KEY"]

    def __call__(self, prompt):
        def call():
            r = self.requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": config.GROQ_MODEL, "temperature": 0,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": prompt}]},
                timeout=60,
            )
            if r.status_code != 200:
                raise GroqError(r.status_code, r.text)
            return r.json()["choices"][0]["message"]["content"].strip()
        return _with_retries(call, "Groq")


class MockLLM:
    """Offline stand-in: returns the top excerpt. Used only to test the pipeline without an API key."""

    model = "mock"

    def __call__(self, prompt):
        first = prompt.split("[1]", 1)[1].split("\n[2]", 1)[0]
        header, _, body = first.partition("\n")
        return f"{body.strip()} [{header.strip(' ()')}]", 0.0


def make_llm():
    return {"mock": MockLLM, "groq": GroqLLM}.get(config.LLM_BACKEND, GeminiLLM)()


class AdvisorAssistant:
    def __init__(self, persist=True):
        self.collection, self.chunks = build_collection(persist=persist)
        self.llm = make_llm()

    def retrieve(self, question, k=config.TOP_K):
        res = self.collection.query(query_texts=[question], n_results=k)
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"id": cid, "text": doc, "score": round(1 - dist, 4), **meta})
        return hits

    def answer(self, question):
        start = time.perf_counter()
        waited = 0.0
        hits = self.retrieve(question)
        top_score = hits[0]["score"] if hits else 0.0

        if top_score < config.ESCALATE_BELOW:
            escalated, reason, text = True, "low retrieval confidence", ESCALATION_MESSAGE
        else:
            context = "\n".join(
                f"[{i}] ({h['policy']}, {h['section']})\n{h['text']}" for i, h in enumerate(hits, 1)
            )
            prompt = f"Policy excerpts:\n{context}\n\nAdvisor question: {question}"
            text, waited = self.llm(prompt)
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
            # Retry waits are excluded so latency reflects the system, not the free-tier queue
            "latency_ms": round((time.perf_counter() - start - waited) * 1000, 1),
            "model": self.llm.model,
        }


if __name__ == "__main__":
    bot = AdvisorAssistant()
    q = "What is the waiting period for cancer under the critical illness plan?"
    print(bot.answer(q))
