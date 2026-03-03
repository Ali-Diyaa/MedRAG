"""
app.py
------
Gradio front-end for the Medical PDF RAG Assistant.

Tabs
----
1. Upload PDF   – parse, chunk, index the report.
2. Summarise    – stream a progressive summary of the first 15 chunks.
3. Chat         – ask questions about the report (with guardrail filter).

Run
---
    python app.py
"""

import threading
import gradio as gr
from transformers import TextIteratorStreamer

# Local modules
from llm import load_chat_llm, load_summ_llm
from pdf_loader import load_and_split
from vector_store import (
    load_embedding_model,
    build_vector_store,
    get_retriever,
    is_medical_query,
)

# ---------------------------------------------------------------------------
# Global state  (populated at upload time)
# ---------------------------------------------------------------------------
state = {
    "docs_split": [],
    "vector_store": None,
    "retriever": None,
    "chat_history": [],
}

# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------
print("Loading embedding model …")
embedding_model = load_embedding_model()

print("Loading chat LLM (medgemma-4b-it) …")
chat_tokenizer, chat_model, chat_pipe = load_chat_llm()

print("Loading summarisation LLM (MediLlama-3.2) …")
summ_tokenizer, summ_model, summ_pipe = load_summ_llm()

# Guardrail classifier – set to None until trained (optional)
# To enable, train via vector_store.train_guardrail() and assign here.
guardrail_clf = None

print("All models ready.\n")


# ---------------------------------------------------------------------------
# Tab 1 – Upload & Process PDF
# ---------------------------------------------------------------------------
def process_pdf(file):
    if file is None:
        return "⚠️ No file uploaded."

    try:
        docs = load_and_split(file.name)
    except Exception as exc:
        return f"❌ Error reading PDF: {exc}"

    if not docs:
        return "⚠️ No readable text found in the PDF."

    try:
        vs = build_vector_store(docs, embedding_model)
        state["docs_split"] = docs
        state["vector_store"] = vs
        state["retriever"] = get_retriever(vs, k=3)
        state["chat_history"] = []
    except Exception as exc:
        return f"❌ Error building vector store: {exc}"

    return f"✅ PDF processed — {len(docs)} chunks indexed."


# ---------------------------------------------------------------------------
# Tab 2 – Progressive summarisation
# ---------------------------------------------------------------------------
def summarize_pdf():
    docs = state["docs_split"]
    if not docs:
        return "⚠️ Upload a PDF first."

    summary_text = ""
    for i, doc in enumerate(docs[:15], start=1):
        prompt = f"Summarize the following medical text concisely:\n{doc.page_content}\n for a non medical user and the summary include only the main issues in that medical report with illustrating to the non medical user, all in 2-3 lines"
        streamer = TextIteratorStreamer(summ_tokenizer, skip_special_tokens=True)
        inputs = summ_tokenizer(prompt, return_tensors="pt").to("cuda")

        def _run(inputs=inputs, streamer=streamer):
            summ_model.generate(
                **inputs,
                max_new_tokens=150,
                streamer=streamer,
                do_sample=False,
            )

        t = threading.Thread(target=_run)
        t.start()
        for new_text in streamer:
            summary_text += new_text
            yield summary_text          # progressive Gradio update
        t.join()


# ---------------------------------------------------------------------------
# Tab 3 – Chat with PDF
# ---------------------------------------------------------------------------
def chat_pdf(user_message: str):
    retriever = state["retriever"]
    chat_history = state["chat_history"]

    if not retriever:
        yield chat_history + [("System", "⚠️ Upload a PDF first!")]
        return

    # --- Guardrail ---------------------------------------------------------
    if guardrail_clf is not None:
        if not is_medical_query(user_message, embedding_model, guardrail_clf):
            reply = (
                "⚠️ This question is outside the medical report context. "
                "I can only answer questions about the uploaded report."
            )
            chat_history.append((user_message, reply))
            yield chat_history
            return

    # --- Retrieve context --------------------------------------------------
    docs = retriever.invoke(user_message)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = (
        "You are a medical AI assistant specialised in explaining medical reports.\n\n"
        "ONLY use the information in the context below.\n"
        "If the answer is not in the context, reply: "
        "'Information not available in the report.'\n\n"
        f"Context:\n{context}\n\n"
        f"Patient Question:\n{user_message}\n\n"
        "Explain clearly in simple medical terms.\n"
    )

    # --- Streamed generation -----------------------------------------------
    streamer = TextIteratorStreamer(chat_tokenizer, skip_special_tokens=True)
    inputs = chat_tokenizer(prompt, return_tensors="pt").to("cuda")

    def _run(inputs=inputs, streamer=streamer):
        chat_model.generate(
            **inputs,
            max_new_tokens=256,
            streamer=streamer,
            do_sample=True,
            temperature=0.3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=chat_tokenizer.eos_token_id,
        )

    t = threading.Thread(target=_run)
    t.start()

    response_text = ""
    for new_text in streamer:
        response_text += new_text
        yield chat_history + [(user_message, response_text)]

    t.join()
    chat_history.append((user_message, response_text))
    state["chat_history"] = chat_history


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="🩺 Medical PDF Assistant") as demo:
    gr.Markdown("# 🩺 Medical PDF Assistant\nUpload a medical report, get a summary, and ask questions about it.")

    # ---- Tab 1 ----
    with gr.Tab("📄 Upload PDF"):
        pdf_file = gr.File(label="Upload Medical PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Process PDF", variant="primary")
        upload_status = gr.Textbox(label="Status", interactive=False)
        upload_btn.click(process_pdf, inputs=[pdf_file], outputs=[upload_status])

    # ---- Tab 2 ----
    with gr.Tab("📝 Summarise PDF"):
        summarize_output = gr.Textbox(
            label="Progressive Summary",
            interactive=False,
            lines=20,
            placeholder="Click 'Summarise' after uploading a PDF …",
        )
        summarize_btn = gr.Button("Summarise", variant="primary")
        summarize_btn.click(
            summarize_pdf, inputs=[], outputs=[summarize_output], queue=True
        )

    # ---- Tab 3 ----
    with gr.Tab("💬 Chat with PDF"):
        chat_output = gr.Chatbot(label="Conversation")
        user_input = gr.Textbox(
            label="Your question",
            placeholder="Ask anything about the uploaded report …",
        )
        send_btn = gr.Button("Send", variant="primary")
        send_btn.click(
            chat_pdf, inputs=[user_input], outputs=[chat_output], queue=True
        )

if __name__ == "__main__":
    demo.launch()
