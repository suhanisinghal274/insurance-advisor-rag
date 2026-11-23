"""Web demo for advisors. In Colab, run:  !python app.py --share"""
import sys

import gradio as gr

from src.rag import AdvisorAssistant

bot = AdvisorAssistant()

EXAMPLES = [
    "What is the waiting period for major cancer under CriticalCare 37?",
    "What is the yearly deductible on HospitalCare Plus Plan C?",
    "How much can a client borrow against a LifeLegacy policy?",
    "Does Lumina offer travel insurance that covers lost luggage?",
]


def ask(question):
    if not question.strip():
        return "Type a question first.", "", ""
    out = bot.answer(question)
    status = ("**Handed off to a human advisor** (" + out["escalation_reason"] + ")") if out["escalated"] else "**Answered from policy documents**"
    sources = "\n".join(f"- {s['policy']} | {s['section']} (score {s['score']:.2f})" for s in out["sources"])
    meta = f"{status}  ·  {out['latency_ms']:.0f} ms  ·  best match score {out['top_score']:.2f}"
    return out["answer"], sources, meta


with gr.Blocks(title="Insurance Advisor Assistant") as demo:
    gr.Markdown("## Insurance Advisor Assistant\nAnswers advisor questions from policy documents with citations, "
                "and hands off to a human when unsure. *Lumina Assurance is fictional; all policies are synthetic.*")
    question = gr.Textbox(label="Advisor question", lines=2)
    btn = gr.Button("Ask", variant="primary")
    meta = gr.Markdown()
    answer = gr.Markdown(label="Answer")
    sources = gr.Markdown(label="Retrieved sections")
    gr.Examples(EXAMPLES, inputs=question)
    btn.click(ask, inputs=question, outputs=[answer, sources, meta])
    question.submit(ask, inputs=question, outputs=[answer, sources, meta])

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv, server_name="0.0.0.0")
