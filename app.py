import gradio as gr

import api_client


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


def chat_fn(message, history, document_id):
    doc_id = document_id.strip() or None
    try:
        result = api_client.ask(message, doc_id)
    except Exception as exc:
        return f"Request to orchestrator failed: {exc}"
    return format_answer(result)


def load_documents():
    docs = api_client.list_documents()
    return [[d["document_id"], d["title"], d["pages"], d["tables"]] for d in docs]


def load_dashboard():
    docs = api_client.list_documents()
    queries = api_client.recent_queries()
    total_tables = sum(d["tables"] for d in docs)
    stats = f"**Indexed documents:** {len(docs)}  \n**Detected tables:** {total_tables}"
    query_rows = [[q["question"], q["latency_ms"], q["valid"]] for q in queries]
    return stats, query_rows


with gr.Blocks(title="LEDGER") as demo:
    gr.Markdown("# LEDGER — Financial Document Intelligence Agent")

    with gr.Tab("Chat"):
        doc_scope = gr.Textbox(label="Scope to document_id (optional)", placeholder="leave empty for corpus-wide")
        gr.ChatInterface(fn=chat_fn, additional_inputs=[doc_scope])

    with gr.Tab("Documents"):
        doc_table = gr.Dataframe(
            headers=["document_id", "title", "pages", "tables"],
            value=load_documents,
        )
        refresh_docs = gr.Button("Refresh")
        refresh_docs.click(fn=load_documents, outputs=doc_table)

    with gr.Tab("Dashboard"):
        stats_md = gr.Markdown()
        query_table = gr.Dataframe(headers=["question", "latency_ms", "valid"])
        refresh_dash = gr.Button("Refresh")

        def _refresh():
            return load_dashboard()

        refresh_dash.click(fn=_refresh, outputs=[stats_md, query_table])
        demo.load(fn=_refresh, outputs=[stats_md, query_table])


if __name__ == "__main__":
    demo.launch()
