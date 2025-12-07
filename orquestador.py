from __future__ import annotations
import gradio as gr
import threading
_PIPELINE_LOCK = threading.Lock()
from generador import responder_consulta
from retriever import top_snippets
from rag_index import Index_coincidencias, traducir_consulta, buscar_info, normalizar_resultados

def pipeline(user_q_original: str, page_size: int = 50, alpha: float = 0.6, top_k: int = 12):
   with _PIPELINE_LOCK:
    # 1) Buscar candidatos en Europe PMC
    user_q= traducir_consulta(user_q_original)
    raw = buscar_info(user_q, page_size=page_size)
    docs = [normalizar_resultados(r) for r in raw if (r.get("title") or r.get("abstractText"))]

    if not docs:
        return "No se pudieron recuperar fuentes para responder la consulta.", "", ""

    # 2) Index local (BM25 + vectores) 
    idx = Index_coincidencias()
    idx.build(docs)

    # 3) Ranking híbrido y selección
    top = idx.search(user_q_original, user_q, top_k=top_k, alpha=alpha)

    # 4) Extraer snippets
    snips = top_snippets(user_q, top, max_snippets=6)

    # 5) Generar respuesta con citas vía Gemini
    
    answer, refs = responder_consulta(user_q_original, snips, model_name="gemma3:1b")



    # 6) Mostrar lista top recuperados
    lst = "\n".join(
        f"- {d.title} — {d.source or 'bioRxiv'} ({d.year or ''}) | DOI:{d.doi or ''} | PDF:{d.pdf or ''}"
        for d, _ in top[:5]
    )
    return user_q, answer, refs, lst

demo = gr.Interface(
    fn=pipeline,
    inputs=[gr.Textbox(label="Consulta")],
    outputs=[
        gr.Textbox(label="query"),
        gr.Markdown(label="Respuesta con citas [n]"),
        gr.Markdown(label="Referencias"),
        gr.Markdown(label="Top recuperados")
    ],
    title="BioRAG",
    description="Write a question in English. For ex. 'What are the pathways associated to TNF proteins?'"
)

if __name__ == "__main__":
    demo.launch()