# ========= rad_index.py

from __future__ import annotations
import requests, re
from dataclasses import dataclass
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from vector_store import VectorStore, _sha256_norm



EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"



# Prompt few-shot, con dos anda bien
_FEWSHOT_EPMC = """
You produce ONE English query for Europe PMC in Lucene-style syntax.
Rules:
- Use AND/OR/NOT, quotes for phrases, parentheses, and * when appropriate.
- You may use fields (TITLE:, ABSTRACT:), but do NOT use SRC: or PUBLISHER: unless explicitly requested.
- Keep gene/protein names and acronyms unchanged (e.g., BRCA1, EGFR, IL-4, COPD).
- Include reasonable synonyms with OR.
- Reply with ONLY the query, on a single line, with no explanations or extra quotes.

Example 1
Question: Does the BRCA1 gene play a role in breast cancer?
Your answer: ("BRCA1" OR "BRCC1") AND ("breast cancer" OR "mammary carcinoma")

Example 2
Question: Genetic causes of Parkinson’s disease.
Your answer: ("Parkinson's disease" OR parkinsonism OR Parkinson) AND (genetic OR heritable OR monogenic OR mutation* OR variant*) AND (gene OR protein)

Now your turn
Question: {question}
Your answer:
""".strip()



def traducir_consulta(
    consulta: str,
    model_name: str = "qwen2.5-coder:1.5b",   
    *,
    temperature: float = 0.1,                # un toque de flexibilidad para sinonimos
    max_tokens: int = 70,
) -> str:
    """
    Dada una consulta, devuelve una QUERY en inglés ya expandida para Europe PMC
    (booleanos, frases). Devuelve SOLO la query en una línea.
    """
    prompt = _FEWSHOT_EPMC.format(question=consulta)
    res = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    text = (res.get("response") or "").strip()

    # Limpieza mínima: primera línea, sin comillas/código/etiquetas
    line = text.splitlines()[0].strip()
    for q in ('"""', "```", "`", '"', "“", "”"):
        if line.startswith(q) and line.endswith(q):
            line = line[len(q):-len(q)].strip()
    if line.lower().startswith("salida:"):
        line = line.split(":", 1)[1].strip()

    return line


STOP = {"and", "or", "not", "AND", "OR", "NOT"}  # sujeto a cambios
def extraer_palabras_clave(
    dsl: str,
    include_words_from_phrases: bool = False,
    extra_stop: Optional[List[str]] = None,
    phrase_mode: str = "as_is",  # "as_is" | "quoted" | "underscore"
) -> str:
    """
    Extrae keywords de una query tipo Lucene/Europe PMC y devuelve un string
    con términos separados por espacio.
    """
    STOP = {"and", "or", "not"}
    if extra_stop:
        STOP |= {w.lower() for w in extra_stop}

    # 1) Frases entre comillas (y spans)
    quoted_matches = list(re.finditer(r'"([^"]+)"', dsl))
    phrases = []
    seen_phrases = set()
    for m in quoted_matches:
        ph = m.group(1).strip()
        lph = ph.lower()
        if lph and lph not in seen_phrases:
            seen_phrases.add(lph)
            phrases.append(ph)

    # 2) Quitar las frases del string para no tokenizarlas como palabras sueltas
    s = dsl
    for m in reversed(quoted_matches):  # reversed para no romper offsets
        s = s[:m.start()] + " " + s[m.end():]

    # 3) Limpiar fields/paréntesis/booleanos
    s = re.sub(r'\b(TITLE|ABSTRACT|AUTHOR|JOURNAL|AFFILIATION|KEYWORD)\s*:\s*', ' ', s, flags=re.I)
    s = re.sub(r'[\(\)]', ' ', s)
    s = re.sub(r'\b(AND|OR|NOT)\b', ' ', s, flags=re.I)

    # 4) Palabras sueltas (conservando dígitos/guiones)
    words_all = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", s)

    # 5) Si no queremos palabras que ya están dentro de frases, filtrarlas
    phrase_word_set = set()
    if not include_words_from_phrases and phrases:
        for ph in phrases:
            phrase_word_set.update(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", ph.lower()))

    out_list = []
    seen = set()

    # Primero, frases (orden de aparición)
    for ph in phrases:
        key = ph.lower()
        if key in seen:
            continue
        seen.add(key)
        if phrase_mode == "quoted":
            out_list.append(f'"{ph}"')
        elif phrase_mode == "underscore":
            out_list.append(ph.replace(" ", "_"))
        else:  # "as_is"
            out_list.append(ph)

    # palabras sueltas
    for w in words_all:
        lw = w.lower()
        if lw in STOP:
            continue
        if not include_words_from_phrases and lw in phrase_word_set:
            continue
        if lw not in seen:
            seen.add(lw)
            out_list.append(w)

    # 6) Un solo string separado por espacios
    return " ".join(out_list)





def buscar_info(query: str, page_size: int = 50):
    # Solo preprints + solo bioRxiv por ahora
    q = f'({query}) AND SRC:PPR AND PUBLISHER:"bioRxiv"'
    #Otros filtros pueden ser SRC: MED (pubmed), PMC(pubmed central), hay otros que no entiendo del todo
    #                         PUBLISHER: bioRxiv, medRxiv, Research Square...
    #                         Elsevier, Springer Nature, PLOS (para articulos revisados por pares)

    params = {"query": q, "format": "json", "pageSize": str(page_size), "resultType": "core"}
    r = requests.get(EPMC_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("resultList", {}).get("result", [])

@dataclass
class Doc:
    id: str
    title: str
    abstract: str
    fulltext: str
    source: str
    doi: str | None
    url: str | None
    pdf: str | None
    year: int | None


def fulltext_de_pmcid(pmcid: str) -> str:
    """Descarga y parsea fulltext XML de Europe PMC si está disponible."""
    try:
        url = EPMC_FULLTEXT.format(pmcid=pmcid)
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return ""
        root = ET.fromstring(r.content)

        # Buscar todos los párrafos <p> y concatenar
        paragraphs = [el.text for el in root.findall(".//p") if el.text]
        return "\n".join(paragraphs)
    except Exception:
        return ""



def normalizar_resultados(r: Dict) -> Doc:
    doi = r.get("doi")
    url = r.get("fullTextUrlList", {}).get("fullTextUrl", [])
    pdf = None
    for u in url:
        if u.get("documentStyle") == "pdf":
            pdf = u.get("url")
            break

    # Fallback típico de bioRxiv si no hay PDF directo en Europe PMC:
    if (not pdf) and doi:
        # A veces Europe PMC no trae el link PDF; el patrón clásico en bioRxiv:
        pdf = f"https://www.biorxiv.org/content/{doi}v1.full.pdf"
    year = int(r["pubYear"]) if r.get("pubYear") and r["pubYear"].isdigit() else None
    
    pmcid = r.get("pmcid")
    fulltext = fulltext_de_pmcid(pmcid) if pmcid else ""

    
    return Doc(
        id=r.get("id") or doi or r.get("pmid") or r.get("ext_id") or r.get("title", "")[:50],
        title=r.get("title",""),
        abstract=r.get("abstractText","") or r.get("abstract",""),
        fulltext=fulltext, 
        source=r.get("source",""),
        doi=doi,
        url=r.get("pdfUrl") or r.get("fullTextUrl") or r.get("authorUrl") or r.get("citedByUrl") or r.get("pmcid"),
        pdf=pdf,
        year=year
    )

class Index_coincidencias:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.emb = SentenceTransformer(embed_model)
        self.docs: List[Doc] = []
        self.bm25 = None
        self.faiss_index = None
        self.doc_vecs = None

    def build(self, docs: List[Doc]):
       # 1) filtrar y preparar textos 
       self.docs = [d for d in docs if (d.title or d.abstract or d.fulltext)]
       texts = [f"{d.title}\n\n{d.abstract}\n\n{d.fulltext}" for d in self.docs]

       # 2) BM25
       tokenized = [t.lower().split() for t in texts]
       self.bm25 = BM25Okapi(tokenized)

       # 3) VectorStore: reusar embeddings si existen; si faltan, calcular y guardar
       vs = VectorStore(
           path="./biorag_store",
           embed_fn=lambda L: self.emb.encode(L, convert_to_numpy=True, normalize_embeddings=True),
       )
   
       # armar lista de chunks para upsert (1 chunk por doc: title+abstract+fulltext)
       chunks = []
       for d, t in zip(self.docs, texts):
           chunks.append({
               "doc_id": getattr(d, "doi", None) or getattr(d, "url", None) or getattr(d, "pdf", None) or getattr(d, "id", None),
               "chunk_id": 0,
               "text": t,
               "source": getattr(d, "source", None) or "EuropePMC",
               "title": getattr(d, "title", None),
           })

       # esto agrega lo nuevo (dedupe por hash de texto normalizado)
       vs.nuevos(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")

       # recuperar embeddings Alineados al orden de self.docs
       hashes = [_sha256_norm(t) for t in texts]
       self.doc_vecs = vs.embeddings_x_hash(hashes)  # shape: (N, dim)

       # 4) FAISS 
       dim = self.doc_vecs.shape[1]
       self.faiss_index = faiss.IndexFlatIP(dim)
       self.faiss_index.add(self.doc_vecs)


    def search(self, query_original: str, query:str, top_k: int = 20, alpha: float = 0.6):
        # alpha:pondero BM25 y embeddings
        q_bm25 = extraer_palabras_clave(query) # de la query en pmc, saco palabras clave para similitud lexica
        q_vec = self.emb.encode([query_original], convert_to_numpy=True, normalize_embeddings=True) #la query original la vectorizo para similitud semantica
        # FAISS
        D, I = self.faiss_index.search(q_vec, top_k)
        vec_scores = D[0]
        # BM25
        bm25_scores = self.bm25.get_scores(q_bm25.lower().split())
        # Combinar
        combined = {}
        for rank, idx in enumerate(I[0]):
            combined[idx] = alpha*vec_scores[rank] + (1-alpha)*(bm25_scores[idx]/max(1.0, np.max(bm25_scores)))
        # top ordenado
        top = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.docs[i], score) for i, score in top]