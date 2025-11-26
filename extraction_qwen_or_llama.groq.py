#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq

PROMPTS = [f"P{i}" for i in range(1, 9)]

# =====================================================================
# Carregamento de artigos (JSON ou JSONL)
# =====================================================================
def load_all_articles(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data_str = f.read().strip()

    if not data_str:
        raise ValueError("Arquivo de entrada está vazio.")

    # JSON (lista ou objeto único)
    try:
        data = json.loads(data_str)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # JSONL
    articles = []
    for line in data_str.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                articles.append(obj)
        except json.JSONDecodeError:
            continue

    if not articles:
        raise ValueError("Não foi possível interpretar o arquivo nem como JSON nem como JSONL.")

    return articles


# =====================================================================
# Heurística para truncar FullText priorizando Results/Figures
# =====================================================================
def selective_truncate(full_text: str, max_chars: int = 12000) -> str:
    if not full_text:
        return ""
    s = full_text.strip()
    if len(s) <= max_chars:
        return s

    # Heurística simples: localizar blocos de "Results" e "Figure"/"Fig."
    candidates: List[str] = []
    # Quebras por cabeçalhos básicos
    parts = re.split(r"\n\s*(?=Abstract|Introduction|Methods|Results|Discussion|Conclusion|Conclusions)\b", s, flags=re.I)
    for part in parts:
        if re.search(r"\bResults\b", part, flags=re.I):
            candidates.append(part)
    # Se não encontrou, tenta blocos com Figure/Fig.
    if not candidates:
        fig_blocks = re.split(r"(?=Figure\s+\d+|Fig\.\s*\d+)", s, flags=re.I)
        for b in fig_blocks:
            if re.search(r"(Figure\s+\d+|Fig\.\s*\d+)", b, flags=re.I):
                candidates.append(b)

    if candidates:
        joined = "\n\n".join(candidates)
        return joined[:max_chars]

    # Fallback: início do texto
    return s[:max_chars]


# =====================================================================
# Build ARTICLE block (template do prompt)
# =====================================================================
def build_article_block(article: Dict[str, Any], max_chars: int) -> str:
    pmid = str(article.get("pmid", "") or "")
    title = article.get("title", "") or ""
    journal = article.get("journal", "") or ""
    year = str(article.get("year", "") or "")
    authors = article.get("authors", []) or []

    first_author = authors[0] if authors else "unspecified"
    other_authors = ", ".join(authors[1:]) if len(authors) > 1 else "unspecified"

    abstract = article.get("abstract", "") or ""
    full_text = article.get("full_text", "") or ""
    fulltext_block = selective_truncate(full_text, max_chars=max_chars)

    lines = []
    lines.append(f"PMID: {pmid}")
    lines.append(f"First author: {first_author}")
    lines.append(f"Other authors: {other_authors}")
    lines.append(f"Title: {title}")
    lines.append(f"Year: {year} | Journal: {journal}")
    lines.append(f"Abstract: {abstract.strip()}")
    lines.append(f"FullText: {fulltext_block}")

    return "\n".join(lines)


# =====================================================================
# Instruções base (system) – exatamente como no prompt
# =====================================================================
BASE_SYSTEM_INSTRUCTIONS = """
Task: Answer strictly based on ARTICLE. No external knowledge. If missing, return {"pmid":"<pmid>","not_found":true}.
Format: Valid JSON only. No text outside JSON.
Citations: For each item, include {quote (<= ~30 words), section, confidence}.
Deduplication: Consolidate synonyms and redundancies. Normalize names as in the article.
Limits: Max 15 items per prompt. Prefer the most central items per the article.
Common fields: not_found (boolean), items (list), citations (list of {quote, section, confidence}), errors (optional).
"""

PROMPT_TEMPLATES = {
    "P1": """ARTICLE
{article}

P1. Transcription factors regulating lignin pathway genes
Objective: Extract TFs, their targets within the lignin/monolignol pathway, and the reported evidence.
Additional instructions:
- evidence_type ∈ {{binding_assay, ChIP, reporter_assay, perturbation, coexpression, genetic_association, other, unspecified}}
- effect ∈ {{upregulates, downregulates, dual, unclear}}
Item schema:
{{
  pmid: string,
  not_found: boolean,
  items: [
    {{
      tf: string,
      target_gene: string,
      species: string or "unspecified",
      evidence_type: string,
      effect: string,
      notes: optional short string
    }}
  ],
  citations: [{{quote, section, confidence}}]
}}
Return only valid JSON, with no additional text.""",

    "P2": """ARTICLE
{article}

P2. Enzymes in the monolignol/lignin pathway and perturbation effects
Objective: List enzymes mentioned and any functional effects reported.
Additional instructions: If knockdown/KO/OE effects exist, summarize impacts on lignin content/composition (S/G), saccharification, mechanics, defense.
Item schema:
{{
  pmid: string,
  not_found: boolean,
  items: [
    {{
      enzyme: string,
      gene_symbol_or_id: string,
      species: string or "unspecified",
      pathway_role: string,
      perturbation_effect: string,
      evidence_type: string,
      notes: optional short string
    }}
  ],
  citations: [{{quote, section, confidence}}]
}}
Return only valid JSON, with no additional text.""",

    "P3": """ARTICLE
{article}

P3. Post-transcriptional regulation (miRNAs, alternative splicing, lncRNAs, RBPs)
Objective: Identify post-transcriptional regulators and their targets/effects.
Additional instructions: mechanism ∈ {{miRNA_cleavage, translational_repression, alternative_splicing, lncRNA_interaction, RBP_binding, other, unspecified}}
Item schema:
{{
  pmid: string,
  not_found: boolean,
  items: [
    {{
      regulator: string,
      mechanism: string,
      target: string,
      species: string or "unspecified",
      observed_effect_on_lignin: string,
      evidence_type: string,
      notes: optional short string
    }}
  ],
  citations: [{{quote, section, confidence}}]
}}
Return only valid JSON, with no additional text.""",

    "P4": """ARTICLE
{article}

P4. Datasets, resources, and analytical methods
Objective: Capture datasets/resources (e.g., AspWood, PGX Atlas) and techniques (RNA‑seq, ChIP, TGMI).
Item schema:
{{
  pmid: string,
  not_found: boolean,
  items: [
    {{
      resource_or_method: string,
      species_or_material: string,
      samples_or_scale: string or "unspecified",
      purpose_in_article: string,
      key_output: string,
      notes: optional short string
    }}
  ],
  citations: [{{quote, section, confidence}}]
}}
Return only valid JSON, with no additional text.""",

    "P5": """ARTICLE
{article}

P5. Limitations, gaps, and next steps
Objective: Record limitations and future directions as stated by the authors.
Item schema:
{{
  pmid: string,
  not_found: boolean,
  items: [
    {{
      limitation_or_gap: string,
      proposed_future_work: string or "unspecified",
      scope: string,
      notes: optional short string
    }}
  ],
  citations: [{{quote, section, confidence}}]
}}
Return only valid JSON, with no additional text.""",

    "P6": """ARTICLE
{article}

P6. Regulatory hierarchy and network architecture
Objective: Describe the hierarchy (e.g., NAC/VND/SND → MYB46/83 → enzymes) and key relations.
Item schema:
{{
  pmid: string,
  not_found: boolean,
  items: [
    {{
      layer: string,
      entity: string,
      relation: string,
      target: string,
      evidence_type: string,
      notes: optional short string
    }}
  ],
  citations: [{{quote, section, confidence}}]
}}
Return only valid JSON, with no additional text.""",

    "P7": """ARTICLE
{article}

P7. Spatial/temporal expression patterns relevant to lignin
Objective: Summarize where/when key genes are expressed (if provided by the article).
Item schema:
{{
  pmid: string,
  not_found: boolean,
  items: [
    {{
      gene_or_tf: string,
      tissue_or_stage: string,
      species: string or "unspecified",
      expression_pattern: string,
      dataset_or_method: string,
      notes: optional short string
    }}
  ],
  citations: [{{quote, section, confidence}}]
}}
Return only valid JSON, with no additional text.""",

    "P8": """ARTICLE
{article}

P8. Unanswerable control (hallucination detection)
Question: "Which floral volatile compounds does the article quantify, and which biosynthetic pathways are analyzed for those volatiles?"
Expected: not_found: true for most lignin/secondary wall articles. If the article genuinely covers floral volatiles, extract normally.
Item schema:
{{
  pmid: string,
  not_found: boolean,
  items: [
    {{
      volatile_compound: string,
      pathway: string,
      species: string,
      measurement_method: string,
      notes: optional short string
    }}
  ],
  citations: [{{quote, section, confidence}}]
}}
Return only valid JSON, with no additional text."""
}


# =====================================================================
# Parsing robusto do JSON retornado pelo modelo
# =====================================================================
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()

    # Blocos ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Primeiro '{' ... último '}'
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        candidate = s[start:end+1]
        # reparos leves
        candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def ensure_response_shape(pmid: str, obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = obj.copy() if isinstance(obj, dict) else {}
    out["pmid"] = str(out.get("pmid", pmid))
    items = out.get("items", [])
    if not isinstance(items, list):
        items = []
    # Limite de 15 itens (regra do prompt)
    items = items[:15]
    out["items"] = items
    citations = out.get("citations", [])
    if not isinstance(citations, list):
        citations = []
    out["citations"] = citations
    nf = out.get("not_found", None)
    if isinstance(nf, bool):
        out["not_found"] = nf
    else:
        out["not_found"] = False if len(items) > 0 else True
    # Normalização leve das citações
    norm_cits = []
    for c in citations[:15]:
        if isinstance(c, dict):
            q = str(c.get("quote", "")).strip()
            sec = str(c.get("section", "") or "Unknown").strip()
            try:
                conf = float(c.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            norm_cits.append({"quote": q[:300], "section": sec or "Unknown", "confidence": conf})
    out["citations"] = norm_cits
    return out


# =====================================================================
# Groq SDK – chamada com retry
# =====================================================================
def call_groq_chat(
    client: Groq,
    model: str,
    system_msg: str,
    user_msg: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
    sleep_between: float = 0.8
) -> Dict[str, Any]:
    last_err: Optional[str] = None

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                # Se suportado pelo modelo/endpoint:
                # response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return {"ok": True, "content": content}
        except Exception as e:
            last_err = f"tentativa {attempt} falhou: {e}"
            time.sleep(sleep_between * attempt)

    return {"ok": False, "error": last_err or "erro desconhecido"}


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Extrai P1–P8 (Groq) e gera JSONL")
    ap.add_argument("--input", "-i", default="articles_with_fulltext.json", help="JSON ou JSONL com artigos")
    ap.add_argument("--output", "-o", default="qwen_or_llama_outputs.jsonl", help="JSONL de saída")
    ap.add_argument("--model", "-m", default="qwen/qwen3-32b", help="Modelo Groq (ex.: qwen/qwen3-32b, llama-3.1-8b-instant, ...)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--max_chars", type=int, default=12000)
    ap.add_argument("--articles_limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Defina GROQ_API_KEY no ambiente: export GROQ_API_KEY='sua-chave'")

    client = Groq(api_key=api_key)

    articles = load_all_articles(args.input)
    if args.articles_limit > 0:
        articles = articles[:args.articles_limit]

    processed = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for art in articles:
            pmid = str(art.get("pmid", "") or "")
            article_block = build_article_block(art, max_chars=args.max_chars)

            responses: Dict[str, Any] = {}
            for pid in PROMPTS:
                user_msg = PROMPT_TEMPLATES[pid].format(article=article_block)
                call = call_groq_chat(
                    client=client,
                    model=args.model,
                    system_msg=BASE_SYSTEM_INSTRUCTIONS,
                    user_msg=user_msg,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    retries=args.retries,
                )
                if not call["ok"]:
                    response_obj = {
                        "pmid": pmid,
                        "not_found": True,
                        "items": [],
                        "citations": [],
                        "errors": f"request_failed: {call['error']}",
                    }
                else:
                    parsed = extract_json_from_text(call["content"])
                    response_obj = ensure_response_shape(pmid, parsed)

                raw_text_str = json.dumps(response_obj, ensure_ascii=False)
                responses[pid] = {"response": response_obj, "raw_text": raw_text_str}

                if args.sleep > 0:
                    time.sleep(args.sleep)

            out_line = {
                "pmid": pmid,
                "prompts": PROMPTS[:],
                "responses": responses
            }
            fout.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            processed += 1

    print(f"Concluído. Artigos processados: {processed}. Saída em {args.output}")


if __name__ == "__main__":
    main()