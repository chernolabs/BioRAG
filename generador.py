# ========= generador.py

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import ollama


def mapea_citas(docs: List[object]) -> Dict[str, tuple[int, object]]:
    out: Dict[str, tuple[int, object]] = {}
    i = 1
    for d in docs:
        key = getattr(d, "doi", None) or getattr(d, "pdf", None) or getattr(d, "url", None) or getattr(d, "id", None)
        if key and key not in out:
            out[key] = (i, d)
            i += 1
    return out

def _genera_refs(cmap: Dict[str, tuple[int, object]]) -> str:
    lines = []
    for k, (num, d) in sorted(cmap.items(), key=lambda x: x[1][0]):
        title = (getattr(d, "title", "") or "").strip()
        src = getattr(d, "source", None) or "bioRxiv"
        year = getattr(d, "year", None)
        link = getattr(d, "pdf", None) or getattr(d, "url", None) or (
            f"https://doi.org/{getattr(d,'doi')}" if getattr(d, "doi", None) else ""
        )
        lines.append(f"[{num}] {title} — {src}{f' ({year})' if year else ''}. {link}")
    return "\n".join(lines)

def crea_contexto_y_citas(question: str, snippets: List[Tuple[object, str]]):
    used_docs: List[object] = [d for (d, _) in snippets]
    cmap = mapea_citas(used_docs)

    def cite_for(d: object) -> str:
        k = getattr(d, "doi", None) or getattr(d, "pdf", None) or getattr(d, "url", None) or getattr(d, "id", None)
        return f"[{cmap[k][0]}]" if k in cmap else ""

    ctx_lines = [f"- {s} {cite_for(d)}" for (d, s) in snippets]
    ctx = "\n".join(ctx_lines)

    system = (

        "You are a biomedical assistant. Respond in English and only based on the provided context."
        "Every claim must include its inline citation [n] as justification. If something is not in the context, state that there is no evidence in the retrieved sources. Answer coherently and cohesively as if speaking to the user."

    )
    user = (
        f"Context (each line finishes with its citation [n]):\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Write a brief, precise answer integrating the citations [n]"
    )
    prompt = system + "\n\n" + user
    refs_md = _genera_refs(cmap)
    return prompt, refs_md



def _ollama_text(model: str, prompt: str, *, host: str | None = None, options: dict | None = None) -> str:
    """
    Funcion auxiliar para que ollama siempre de una respuesta
    Intenta 1) generate(stream=False), 2) chat() y extrae texto de dict/objeto/stream.
    Lanza RuntimeError si ambas rutas fallan.
    """
    options = options or {}
    client = ollama.Client(host=host) if host else None

    # 1) generate(stream=False)
    try:
        gen = (client.generate if client else ollama.generate)(
            model=model, prompt=prompt, options=options, stream=False 
        )
        # dict clásico
        if isinstance(gen, dict):
            txt = gen.get("response") or (gen.get("message") or {}).get("content") or ""
            if txt:
                return str(txt).strip()
        # objeto con .response o .message.content
        txt = getattr(gen, "response", None)
        if not txt:
            msg = getattr(gen, "message", None)
            if isinstance(msg, dict):
                txt = msg.get("content", "")
            else:
                txt = getattr(msg, "content", "") if msg is not None else ""
        if txt:
            return str(txt).strip()
        # stream/generador (por las dudas)
        if hasattr(gen, "__iter__") and not isinstance(gen, (str, bytes, dict)):
            out = []
            for ch in gen:
                if isinstance(ch, dict) and "response" in ch:
                    out.append(ch["response"])
                elif hasattr(ch, "response"):
                    out.append(getattr(ch, "response") or "")
            if out:
                return "".join(out).strip()
    except Exception:
        pass  # seguimos al plan B

    # 2) chat()
    try:
        chat = (client.chat if client else ollama.chat)(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=options
        )
        if isinstance(chat, dict):
            return (chat.get("message") or {}).get("content", "") or ""
        msg = getattr(chat, "message", None)
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
        return getattr(msg, "content", "") or ""
    except Exception as e:
        raise RuntimeError(f"Ollama falló en generate/chat: {e}")





def responder_consulta(
    question: str,
    snippets: List[Tuple[object, str]],
    model_name: str = "gemma3:1b",
    *,
    host: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 300,
) -> Tuple[str, str]:
    """
    Devuelve (answer, refs_md). 
    Usa _ollama_text() para obtener el texto .
    """
    if not snippets:
        return "No se pudieron recuperar fuentes para responder la consulta.", ""

    # 1) Construcción del prompt + refs
    try:
        prompt, refs_md = crea_contexto_y_citas(question, snippets)
    except Exception as e:
        bullets = "\n".join(f"- {s}" for _, s in snippets)
        ans = (f"No se pudo preparar el prompt ({type(e).__name__}: {e}).\n\n"
               f"Fragmentos relevantes:\n\n{bullets}")
        return ans, ""

    # 2) Llamada al modelo (vía helper robusto)
    try:
        text = _ollama_text(
            model=model_name,
            prompt=prompt,
            host=host,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        text = (text or "").strip()
        if not text:
            text = "No se obtuvo texto del modelo local."
        return text, refs_md

    except Exception as e:
        bullets = "\n".join(f"- {s}" for _, s in snippets)
        ans = (f"No se pudo generar con el modelo local. ({type(e).__name__}: {e})\n\n"
               f"Fragmentos relevantes:\n\n{bullets}")
        return ans, refs_md

