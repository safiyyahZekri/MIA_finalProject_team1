import time

import gradio as gr

import api_client
from evidence import render_citation_page


def format_answer(result: dict) -> str:
    answer_type = result["answer_type"]
    params = result["params"]
    evidence = result["evidence"]

    if answer_type == "direct":
        body = f"**Answer:** {params['value']}"
    elif answer_type == "calculated":
        body = f"**Answer:** {params['value']}\n\n**Formula:** `{params['formula']}`"
    elif answer_type == "multi_span":
        items = "\n".join(f"- {v}" for v in params["values"])
        body = f"**Answer:**\n{items}"
    else:
        return f"**I don't have enough evidence to answer this.**\n\n_{params['reason']}_"

    if evidence:
        citations = "\n".join(
            f"- {e['document_id']}, page {e['page']}" + (f" ({e['section']})" if e.get("section") else "")
            for e in evidence
        )
        body += f"\n\n**Sources:**\n{citations}"
    return body


def answer_note(result: dict, seconds: float) -> str:
    """How the answer was produced, so an answer-cache hit is visible."""
    if result.get("cache_hit"):
        match = (
            "a reworded question" if result.get("cache_match") == "reworded" else "the same question"
        )
        return (
            f"\n\n_Answered from the answer cache ({match} was asked before) "
            f"in {seconds * 1000:.0f} ms, without running the agent._"
        )
    return f"\n\n_Answered by the agent in {seconds:.1f} s._"


def chat_fn(message, history, document_id):
    doc_id = document_id.strip() or None
    try:
        result = api_client.ask(message, doc_id)
    except Exception as exc:
        return f"Request to orchestrator failed: {exc}"
    return format_answer(result)


def answer_with_evidence(message, history, document_id):
    """Answer a question and render every citation box on its first source page."""
    history = history or []
    doc_id = document_id.strip() or None
    try:
        started = time.perf_counter()
        result = api_client.ask(message, doc_id)
        answer = format_answer(result) + answer_note(result, time.perf_counter() - started)
    except Exception as exc:
        answer = f"Request to orchestrator failed: {exc}"
        return (
            "",
            [
                *history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ],
            None,
            "",
            None,
        )

    review_context = {
        "question": message,
        "document_id": doc_id,
        "original_answer": result,
    }

    # Boxes arrive beside the answer; its citations keep the strict schema.
    evidence = [item for item in result.get("evidence_boxes") or [] if item.get("bbox")]
    if not evidence:
        caption = "No bounding-box coordinates were returned for this answer."
        return (
            "",
            [
                *history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ],
            None,
            caption,
            review_context,
        )

    first = evidence[0]
    same_page = [
        item["bbox"]
        for item in evidence
        if item["document_id"] == first["document_id"] and item["page"] == first["page"]
    ]
    try:
        pdf_bytes = api_client.download_document(first["document_id"])
        highlighted = render_citation_page(pdf_bytes, first["page"], same_page)
        caption = (
            f"Highlighted evidence: {first['document_id']}, page {first['page']}"
            + (f", {first['section']}" if first.get("section") else "")
        )
    except Exception as exc:
        highlighted = None
        caption = f"The citation is valid, but its source PDF could not be rendered: {exc}"
    return (
        "",
        [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ],
        highlighted,
        caption,
        review_context,
    )


def submit_review(review_context, verdict, corrected_answer, comment):
    if not review_context:
        return "Ask a question before submitting a review."
    verdict_value = {
        "Correct": "correct",
        "Needs correction": "needs_correction",
    }.get(verdict)
    if verdict_value is None:
        return "Choose whether the answer is correct or needs correction."
    correction = (corrected_answer or "").strip() or None
    if verdict_value == "needs_correction" and correction is None:
        return "Enter the corrected answer before submitting."
    try:
        record = api_client.submit_review(
            question=review_context["question"],
            original_answer=review_context["original_answer"],
            verdict=verdict_value,
            corrected_answer=correction,
            comment=(comment or "").strip() or None,
            document_id=review_context.get("document_id"),
        )
    except Exception as exc:
        return f"Review could not be saved: {exc}"
    return (
        f"Review saved with ID `{record['review_id']}`. "
        "It is recorded as human feedback and does not bypass answer validation."
    )


def upload_pdf(file_path):
    if not file_path:
        return "Choose a PDF first.", ""
    try:
        result = api_client.ingest_pdf(file_path)
    except Exception as exc:
        return f"Upload failed: {exc}", ""
    return (
        f"Indexed **{result['filename']}** ({result['chunks_indexed']} chunks).",
        result["document_id"],
    )


def load_documents():
    docs = api_client.list_documents()
    return [[d["document_id"], d["filename"], d["pages"], d["tables"]] for d in docs]


def load_dashboard():
    docs = api_client.list_documents()
    queries = api_client.recent_queries()
    reviews = api_client.list_reviews()
    total_tables = sum(d["tables"] for d in docs)
    stats = (
        f"**Indexed documents:** {len(docs)}  \n"
        f"**Detected tables:** {total_tables}  \n"
        f"**Human reviews:** {len(reviews)}"
    )
    query_rows = [
        [q["question"], q["latency_ms"], q["valid"], q.get("cache") or ""] for q in queries
    ]
    review_rows = [
        [
            review["review_id"],
            review["verdict"],
            review["question"],
            review.get("corrected_answer") or "",
            review["created_at"],
        ]
        for review in reviews
    ]
    return stats, query_rows, review_rows


def _correction_rows(records):
    return [
        [
            record["correction_id"],
            record["chunk_id"],
            record["corrected_by"],
            record["created_at"],
            record.get("comment") or "",
        ]
        for record in records
    ]


def load_extracted_fields(document_id):
    document_id = (document_id or "").strip()
    if not document_id:
        return [], {}, gr.Dropdown(choices=[], value=None), "", [], "Enter a document_id."
    try:
        fields = api_client.get_extracted_fields(document_id)
        corrections = api_client.list_extraction_corrections(document_id)
    except Exception as exc:
        return (
            [],
            {},
            gr.Dropdown(choices=[], value=None),
            "",
            [],
            f"Could not load extracted fields: {exc}",
        )
    if not fields:
        return [], {}, gr.Dropdown(choices=[], value=None), "", [], "No extracted fields found."
    field_map = {field["chunk_id"]: field for field in fields}
    choices = [
        (
            f"p.{field['page']} · {field['content_type']} · "
            f"{field['section']} · {field['content'][:70]}",
            field["chunk_id"],
        )
        for field in fields
    ]
    selected = fields[0]["chunk_id"]
    overview = [
        [
            field["chunk_id"],
            field["page"],
            field["content_type"],
            field["section"],
            field["content"],
        ]
        for field in fields
    ]
    return (
        overview,
        field_map,
        gr.Dropdown(choices=choices, value=selected),
        fields[0]["content"],
        _correction_rows(corrections),
        f"Loaded {len(fields)} extracted fields. Select one below to edit it.",
    )


def select_extracted_field(chunk_id, field_map):
    field = (field_map or {}).get(chunk_id)
    return field["content"] if field else ""


def save_extraction_correction(
    document_id, chunk_id, corrected_text, corrected_by, comment
):
    document_id = (document_id or "").strip()
    corrected_text = (corrected_text or "").strip()
    corrected_by = (corrected_by or "").strip()
    if not document_id or not chunk_id:
        return "Load and select an extracted field first.", []
    if not corrected_text:
        return "Corrected text cannot be empty.", []
    if not corrected_by:
        return "Enter your name or team identifier.", []
    try:
        record = api_client.submit_extraction_correction(
            document_id=document_id,
            chunk_id=chunk_id,
            corrected_text=corrected_text,
            corrected_by=corrected_by,
            comment=(comment or "").strip() or None,
        )
        corrections = api_client.list_extraction_corrections(document_id)
    except Exception as exc:
        return f"Correction could not be applied: {exc}", []
    return (
        f"Correction `{record['correction_id']}` applied and the retrieval index was rebuilt.",
        _correction_rows(corrections),
    )


with gr.Blocks(title="LEDGER") as demo:
    gr.Markdown("# LEDGER — Financial Document Intelligence Agent")

    with gr.Tab("Chat"):
        review_context = gr.State()
        doc_scope = gr.Textbox(label="Scope to document_id (optional)", placeholder="leave empty for corpus-wide")
        chatbot = gr.Chatbot(label="Answers", type="messages")
        question = gr.Textbox(label="Question", placeholder="Ask about the indexed reports")
        ask_button = gr.Button("Ask", variant="primary")
        evidence_image = gr.Image(label="Cited evidence", interactive=False)
        evidence_caption = gr.Markdown()
        ask_button.click(
            fn=answer_with_evidence,
            inputs=[question, chatbot, doc_scope],
            outputs=[
                question,
                chatbot,
                evidence_image,
                evidence_caption,
                review_context,
            ],
        )
        question.submit(
            fn=answer_with_evidence,
            inputs=[question, chatbot, doc_scope],
            outputs=[
                question,
                chatbot,
                evidence_image,
                evidence_caption,
                review_context,
            ],
        )

        with gr.Accordion("Review or correct this answer", open=False):
            review_verdict = gr.Radio(
                ["Correct", "Needs correction"], label="Your review"
            )
            corrected_answer = gr.Textbox(
                label="Corrected answer",
                placeholder="Required when the answer needs correction",
            )
            review_comment = gr.Textbox(
                label="Comment (optional)",
                placeholder="Explain what was wrong or why the correction is supported",
            )
            review_button = gr.Button("Save review")
            review_status = gr.Markdown()
            review_button.click(
                fn=submit_review,
                inputs=[
                    review_context,
                    review_verdict,
                    corrected_answer,
                    review_comment,
                ],
                outputs=review_status,
            )

    with gr.Tab("Upload"):
        pdf_upload = gr.File(label="Financial report PDF", file_types=[".pdf"], type="filepath")
        upload_button = gr.Button("Process and index", variant="primary")
        upload_status = gr.Markdown()
        uploaded_document_id = gr.Textbox(label="Indexed document_id")
        upload_button.click(
            fn=upload_pdf,
            inputs=pdf_upload,
            outputs=[upload_status, uploaded_document_id],
        )

    with gr.Tab("Documents"):
        doc_table = gr.Dataframe(
            headers=["document_id", "filename", "pages", "tables"],
        )
        refresh_docs = gr.Button("Refresh")
        refresh_docs.click(fn=load_documents, outputs=doc_table)
        demo.load(fn=load_documents, outputs=doc_table)

    with gr.Tab("Correct OCR"):
        gr.Markdown(
            "Review extracted text/table chunks, fix an OCR value, and re-index it "
            "without processing the PDF again. Every change keeps an audit record."
        )
        correction_document_id = gr.Textbox(
            label="Document ID", placeholder="Paste an indexed document_id"
        )
        load_fields_button = gr.Button("Load extracted fields", variant="primary")
        correction_status = gr.Markdown()
        extracted_fields_state = gr.State({})
        extracted_fields_table = gr.Dataframe(
            headers=["chunk_id", "page", "type", "section", "extracted content"],
            interactive=False,
            wrap=True,
        )
        field_selector = gr.Dropdown(label="Extracted field")
        corrected_field_text = gr.Textbox(
            label="Corrected extracted text", lines=8
        )
        corrected_by = gr.Textbox(
            label="Corrected by", placeholder="Name or team identifier"
        )
        correction_comment = gr.Textbox(
            label="Reason / source check (optional)"
        )
        save_correction_button = gr.Button("Save correction and re-index")
        correction_history = gr.Dataframe(
            headers=["correction_id", "chunk_id", "corrected_by", "when", "comment"],
            interactive=False,
        )
        load_fields_button.click(
            fn=load_extracted_fields,
            inputs=correction_document_id,
            outputs=[
                extracted_fields_table,
                extracted_fields_state,
                field_selector,
                corrected_field_text,
                correction_history,
                correction_status,
            ],
        )
        field_selector.change(
            fn=select_extracted_field,
            inputs=[field_selector, extracted_fields_state],
            outputs=corrected_field_text,
        )
        save_correction_button.click(
            fn=save_extraction_correction,
            inputs=[
                correction_document_id,
                field_selector,
                corrected_field_text,
                corrected_by,
                correction_comment,
            ],
            outputs=[correction_status, correction_history],
        )

    with gr.Tab("Dashboard"):
        stats_md = gr.Markdown()
        query_table = gr.Dataframe(headers=["question", "latency_ms", "valid", "cache"])
        review_table = gr.Dataframe(
            headers=[
                "review_id",
                "verdict",
                "question",
                "corrected_answer",
                "created_at",
            ]
        )
        refresh_dash = gr.Button("Refresh")

        def _refresh():
            return load_dashboard()

        refresh_dash.click(fn=_refresh, outputs=[stats_md, query_table, review_table])
        demo.load(fn=_refresh, outputs=[stats_md, query_table, review_table])


if __name__ == "__main__":
    demo.launch()
