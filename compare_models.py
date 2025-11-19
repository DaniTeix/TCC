#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models.py

Comparação entre dois modelos (Modelo A = Qwen; Modelo B = Llama)
usando arquivos .jsonl alinhados por prompt (P1–P8), no formato gerado por
rebuild_jsonl_with_rawtext.py:
  { "pmid": "...", "prompts": [...], "responses": {
      "P1": {"response": {...}, "raw_text": "..."}, ..., "P8": {...}
    }
  }

Saída:
- results/figures/*.png: heatmaps (Jaccard, F1), barras (contagem, cobertura, alucinação) e sobreposição (equivalente a Venn) para P1–P3
- results/tables/*.csv: tabelas comparativas (P1–P3) para entidades e relações (A, B, interseção, exclusivos)
- results/metrics_*.csv: métricas agregadas
- results/summary.md: resumo textual

Requisitos:
- Python 3.8+
- pandas, numpy, matplotlib, seaborn

Autor: (você)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict

import warnings
# Suprimir warning de longdouble do NumPy em alguns ambientes
warnings.filterwarnings(
    "ignore",
    message="Signature .*longdouble.*",
    category=UserWarning
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Configurações e constantes
# ----------------------------
PROMPTS = [f"P{i}" for i in range(1, 9)]
PROMPT_DESCRIPTIONS = {
    "P1": "TFs → alvos da via de lignina",
    "P2": "Enzimas estruturais e efeitos de perturbações",
    "P3": "Regulação pós-transcricional",
    "P4": "Métodos e datasets",
    "P5": "Limitações e perspectivas",
    "P6": "Hierarquia regulatória (NAC → MYB → enzimas)",
    "P7": "Padrões espaciais e temporais",
    "P8": "Controle negativo (not_found esperado)",
}

MODEL_A_LABEL = "Modelo A = Qwen"
MODEL_B_LABEL = "Modelo B = Llama"

# Normalização de efeitos para relações
EFFECT_MAP = {
    "up": "upregulates",
    "upregulate": "upregulates",
    "upregulated": "upregulates",
    "upregulates": "upregulates",
    "activate": "upregulates",
    "activates": "upregulates",
    "down": "downregulates",
    "downregulate": "downregulates",
    "downregulated": "downregulates",
    "downregulates": "downregulates",
    "repress": "downregulates",
    "represses": "downregulates",
    "unclear": "unclear",
    "none": "none",
}


# ----------------------------
# Utilidades de normalização
# ----------------------------
def normalize_text(s: Optional[str]) -> str:
    """Normaliza strings:
    - Trim, espaços simples
    - Lowercase
    - Converte setas ↑/↓ para 'up'/'down'
    - Mantém letras/números e alguns sinais (-_/().)
    """
    if s is None:
        return ""
    s0 = str(s)
    s0 = s0.replace("↑", " up ").replace("↓", " down ").replace("→", " -> ")
    s0 = re.sub(r"\s+", " ", s0).strip().lower()
    s0 = re.sub(r"[^a-z0-9\-\_\./\s\(\)]", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0


def normalize_effect(effect: Optional[str]) -> str:
    e = normalize_text(effect)
    for k, v in EFFECT_MAP.items():
        if re.search(rf"\b{k}\b", e):
            return v
    return e or "regulates"


def safe_get(d: Dict, key: str, default=None):
    return d.get(key, default)


# ----------------------------
# Carregamento de dados
# ----------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Carrega um .jsonl onde cada linha é um JSON completo."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                print(f"[WARN] Linha {i} inválida em {path.name}: {e}")
                continue
            records.append(obj)
    return records


def load_article_texts(path: Optional[Path]) -> Dict[str, str]:
    """Carrega textos de artigos (pmid -> texto), opcional, para checagem de 'alucinação'."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


# ----------------------------
# Extração de itens, entidades e relações
# ----------------------------
def extract_items_and_citations(record: Dict[str, Any], prompt: str) -> Tuple[List[Dict[str, Any]], List[str], bool, str]:
    """Extrai itens, citações e flag not_found de um registro por prompt.
    Retorna (items, quotes, not_found, raw_text_str).
    """
    resp_block = record.get("responses", {}).get(prompt, {})
    response = resp_block.get("response", {}) or {}
    items = response.get("items", []) or []
    citations = response.get("citations", []) or []
    quotes = [c.get("quote", "") for c in citations if isinstance(c, dict)]
    not_found = bool(response.get("not_found", False))
    raw_text_str = resp_block.get("raw_text", "") or ""
    return items, quotes, not_found, raw_text_str


def infer_regulator_type(regulator: str, mechanism: str) -> str:
    reg = normalize_text(regulator)
    mech = normalize_text(mechanism)
    if "mirna" in reg or "mi rna" in reg or "mir" in reg or "mirna" in mech or "mi rna" in mech:
        return "miRNA"
    if "lncrna" in reg or "lnc rna" in reg or "lncrna" in mech or "lnc rna" in mech:
        return "lncRNA"
    if "splicing" in reg or "alternative splicing" in reg or "splicing" in mech:
        return "splicing_factor"
    return "regulator"


def extract_entities_relations_per_item(prompt: str, item: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """A partir de um item do prompt, extrai:
    - Entities: lista de pares (entity_norm, type)
    - Relations: lista de triplas (src_norm, relation_norm, tgt_norm)
    """
    entities = []
    relations = []

    if prompt == "P1":
        tf = normalize_text(safe_get(item, "tf"))
        tg = normalize_text(safe_get(item, "target_gene"))
        effect = normalize_effect(safe_get(item, "effect") or "regulates")
        if tf:
            entities.append((tf, "TF"))
        if tg:
            entities.append((tg, "gene"))
        if tf and tg:
            relations.append((tf, effect, tg))

    elif prompt == "P2":
        enz = normalize_text(safe_get(item, "enzyme"))
        gene_id = normalize_text(safe_get(item, "gene_symbol_or_id"))
        effect = normalize_effect(safe_get(item, "perturbation_effect") or "affects")
        if enz:
            entities.append((enz, "enzyme"))
        if gene_id:
            entities.append((gene_id, "gene"))
        # Relação padronizada: enz -> effect -> lignin (alvo constante para comparabilidade)
        if enz:
            relations.append((enz, effect, "lignin"))

    elif prompt == "P3":
        regulator = normalize_text(safe_get(item, "regulator"))
        mechanism = normalize_text(safe_get(item, "mechanism"))
        target = normalize_text(safe_get(item, "target"))
        rtype = infer_regulator_type(regulator, mechanism)
        if regulator:
            entities.append((regulator, rtype))
        if target:
            entities.append((target, "target"))
        if regulator and target:
            rel_label = mechanism or "regulates"
            relations.append((regulator, rel_label, target))

    elif prompt == "P4":
        method = normalize_text(safe_get(item, "resource_or_method"))
        material = normalize_text(safe_get(item, "species_or_material"))
        purpose = normalize_text(safe_get(item, "purpose_in_article"))
        if method:
            entities.append((method, "method"))
        if material:
            entities.append((material, "species_or_material"))
        if method and purpose:
            relations.append((method, "used_for", purpose))

    elif prompt == "P5":
        gap = normalize_text(safe_get(item, "limitation_or_gap"))
        future = normalize_text(safe_get(item, "proposed_future_work"))
        if gap:
            entities.append((gap, "limitation"))
        if future:
            entities.append((future, "future_work"))
        if gap and future:
            relations.append((gap, "addresses", future))

    elif prompt == "P6":
        entity = normalize_text(safe_get(item, "entity"))
        relation = normalize_effect(safe_get(item, "relation") or "regulates")
        target = normalize_text(safe_get(item, "target"))
        if entity:
            entities.append((entity, "regulator"))
        if target:
            entities.append((target, "target"))
        if entity and target:
            relations.append((entity, relation, target))

    elif prompt == "P7":
        gene_tf = normalize_text(safe_get(item, "gene_or_tf"))
        tissue = normalize_text(safe_get(item, "tissue_or_stage"))
        if gene_tf:
            entities.append((gene_tf, "gene_or_tf"))
        if tissue:
            entities.append((tissue, "tissue"))
        if gene_tf and tissue:
            relations.append((gene_tf, "expression_in", tissue))

    elif prompt == "P8":
        # controle negativo
        pass

    else:
        # fallback genérico
        for k, v in item.items():
            val = normalize_text(str(v))
            if val:
                entities.append((val, f"field:{k}"))

    return entities, relations


def aggregate_by_prompt(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Agrega entidades e relações por prompt ao longo de todos os artigos.
    Retorna:
      prompt -> {
        "entities_by_type": Dict[type, Set[str]],
        "relations": Set[(src, rel, tgt)],
        "items_per_pmid": Dict[pmid, List[items_raw]],
        "citations_per_pmid": Dict[pmid, List[quote]],
        "raw_text_per_pmid": Dict[pmid, raw_text_str],
        "not_found_flags": List[bool]
      }
    """
    out = {}
    for p in PROMPTS:
        out[p] = {
            "entities_by_type": defaultdict(set),
            "relations": set(),
            "items_per_pmid": defaultdict(list),
            "citations_per_pmid": defaultdict(list),
            "raw_text_per_pmid": {},
            "not_found_flags": [],
        }

    for rec in records:
        pmid = str(rec.get("pmid", "NA"))
        for p in PROMPTS:
            items, quotes, not_found, raw_text_str = extract_items_and_citations(rec, p)
            out[p]["not_found_flags"].append(not_found)
            if raw_text_str:
                out[p]["raw_text_per_pmid"][pmid] = raw_text_str
            if quotes:
                out[p]["citations_per_pmid"][pmid].extend(quotes)
            if items:
                out[p]["items_per_pmid"][pmid].extend(items)
            for it in items:
                ents, rels = extract_entities_relations_per_item(p, it)
                for e, etype in ents:
                    if e:
                        out[p]["entities_by_type"][etype].add(e)
                for r in rels:
                    if all(r):
                        out[p]["relations"].add(tuple(r))
    return out


# ----------------------------
# Métricas
# ----------------------------
def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if len(a) == 0 and len(b) == 0:
        return np.nan
    return float(len(a & b)) / float(len(a | b)) if (a or b) else np.nan

# Alias (caso você use jacc em algum ponto)
def jacc(a: Set[Any], b: Set[Any]) -> float:
    return jaccard(a, b)


def f1_symmetric(a: Set[Any], b: Set[Any]) -> float:
    denom = len(a) + len(b)
    if denom == 0:
        return np.nan
    return 2.0 * len(a & b) / denom


def coverage(a: Set[Any], b: Set[Any]) -> Tuple[float, float]:
    u = a | b
    if len(u) == 0:
        return (np.nan, np.nan)
    return (len(a) / len(u), len(b) / len(u))


def choose_item_keys_for_hallucination(prompt: str) -> List[str]:
    if prompt == "P1":
        return ["tf", "target_gene"]
    if prompt == "P2":
        return ["enzyme"]
    if prompt == "P3":
        return ["regulator", "target"]
    if prompt == "P6":
        return ["entity", "target"]
    if prompt == "P7":
        return ["gene_or_tf"]
    return []


def compute_hallucination_rate(
    agg: Dict[str, Any],
    article_texts: Dict[str, str],
    prompt: str
) -> Dict[str, float]:
    """Taxa de 'alucinação' (heurística).
    Um item é 'encontrado' se suas chaves principais aparecem no texto do artigo (se fornecido)
    ou no fallback composto por citações + raw_text do próprio prompt.
    """
    keys = choose_item_keys_for_hallucination(prompt)
    if not keys:
        return {"hallucination_rate": np.nan, "n_items": 0, "n_not_found": 0}

    items_per_pmid = agg["items_per_pmid"]
    quotes_per_pmid = agg["citations_per_pmid"]
    raw_text_per_pmid = agg["raw_text_per_pmid"]

    n_items = 0
    n_not_found = 0

    for pmid, items in items_per_pmid.items():
        base_text = article_texts.get(pmid, "")
        if not base_text:
            # fallback: citações + raw_text (que é o JSON do response em string)
            base_text = " ".join(quotes_per_pmid.get(pmid, [])) + " " + raw_text_per_pmid.get(pmid, "")
        base_text_norm = normalize_text(base_text)

        for it in items:
            n_items += 1
            present = True
            for k in keys:
                val = normalize_text(str(it.get(k, "")))
                if not val:
                    continue
                if val not in base_text_norm:
                    present = False
                    break
            if not present:
                n_not_found += 1

    rate = (n_not_found / n_items) if n_items > 0 else np.nan
    return {"hallucination_rate": rate, "n_items": n_items, "n_not_found": n_not_found}


# ----------------------------
# Tabelas comparativas (P1–P3)
# ----------------------------
def prepare_tables_for_prompt(
    prompt: str,
    model_a_agg: Dict[str, Any],
    model_b_agg: Dict[str, Any],
    out_tables_dir: Path
) -> Dict[str, Path]:
    """Gera CSVs comparativos (entidades e relações) para P1–P3."""
    saved = {}

    def flatten_entities(agg):
        rows = []
        for etype, s in agg["entities_by_type"].items():
            for e in sorted(s):
                rows.append({"entity": e, "type": etype})
        return pd.DataFrame(rows)

    dfA_e = flatten_entities(model_a_agg)
    dfB_e = flatten_entities(model_b_agg)

    setA_e = set((r["entity"], r["type"]) for r in dfA_e.to_dict("records")) if len(dfA_e) else set()
    setB_e = set((r["entity"], r["type"]) for r in dfB_e.to_dict("records")) if len(dfB_e) else set()

    inter_e = sorted(list(setA_e & setB_e))
    onlyA_e = sorted(list(setA_e - setB_e))
    onlyB_e = sorted(list(setB_e - setA_e))

    def to_df(pairs):
        return pd.DataFrame([{"entity": e, "type": t} for e, t in pairs])

    setA_r = set(model_a_agg["relations"])
    setB_r = set(model_b_agg["relations"])
    inter_r = sorted(list(setA_r & setB_r))
    onlyA_r = sorted(list(setA_r - setB_r))
    onlyB_r = sorted(list(setB_r - setA_r))

    def rel_df(triples):
        return pd.DataFrame([{"source": s, "relation": r, "target": t} for (s, r, t) in triples])

    outputs = [
        (f"{prompt}_entities_modelA.csv", dfA_e),
        (f"{prompt}_entities_modelB.csv", dfB_e),
        (f"{prompt}_entities_intersection.csv", to_df(inter_e)),
        (f"{prompt}_entities_exclusive_modelA.csv", to_df(onlyA_e)),
        (f"{prompt}_entities_exclusive_modelB.csv", to_df(onlyB_e)),
        (f"{prompt}_relations_modelA.csv", rel_df(setA_r)),
        (f"{prompt}_relations_modelB.csv", rel_df(setB_r)),
        (f"{prompt}_relations_intersection.csv", rel_df(inter_r)),
        (f"{prompt}_relations_exclusive_modelA.csv", rel_df(onlyA_r)),
        (f"{prompt}_relations_exclusive_modelB.csv", rel_df(onlyB_r)),
    ]

    out_tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs:
        path = out_tables_dir / name
        df.to_csv(path, index=False)
        saved[name] = path

    return saved


# ----------------------------
# Visualizações
# ----------------------------
def style_axes(ax):
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def add_footnote(fig, text: str):
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=9, color="dimgray")


def plot_jaccard_heatmaps(jacc_entities: Dict[str, float], jacc_relations: Dict[str, float], outdir: Path):
    prompts = PROMPTS
    df_e = pd.DataFrame({"prompt": prompts, "jaccard": [jacc_entities.get(p, np.nan) for p in prompts]})
    df_r = pd.DataFrame({"prompt": prompts, "jaccard": [jacc_relations.get(p, np.nan) for p in prompts]})

    for title, df, fname in [
        ("Heatmap Jaccard (Entidades) por Prompt", df_e, "heatmap_jaccard_entities.png"),
        ("Heatmap Jaccard (Relações) por Prompt", df_r, "heatmap_jaccard_relations.png"),
    ]:
        plt.figure(figsize=(10, 2.5))
        pivot = df.set_index("prompt").T
        sns.heatmap(pivot, annot=True, cmap="Blues", vmin=0, vmax=1, cbar=True, fmt=".2f")
        plt.title(f"{title}\n{MODEL_A_LABEL} vs {MODEL_B_LABEL}")
        plt.yticks(rotation=0)
        add_footnote(plt.gcf(), "Métrica: Similaridade de Jaccard A vs B por prompt. NaN indica ambos vazios.")
        plt.tight_layout()
        out_path = outdir / fname
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_entity_counts_bar(counts_A: Dict[str, int], counts_B: Dict[str, int], prompts: List[str], outdir: Path, fname: str):
    rows = []
    for p in prompts:
        rows.append({"prompt": p, "modelo": MODEL_A_LABEL, "count": counts_A.get(p, 0)})
        rows.append({"prompt": p, "modelo": MODEL_B_LABEL, "count": counts_B.get(p, 0)})
    df = pd.DataFrame(rows)
    if df["count"].sum() == 0:
        print("[WARN] Contagens de entidades zeradas; pulando gráfico de barras.")
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="prompt", y="count", hue="modelo")
    plt.title(f"Quantidade de Entidades por Prompt (P1–P3)\n{MODEL_A_LABEL} vs {MODEL_B_LABEL}")
    plt.xlabel("Prompt")
    plt.ylabel("Número de entidades únicas")
    plt.legend(title="Modelos")
    style_axes(plt.gca())
    add_footnote(plt.gcf(), "Total de entidades únicas extraídas por prompt (P1–P3).")
    plt.tight_layout()
    out_path = outdir / fname
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_overlap_equivalent(a_only: int, both: int, b_only: int, prompt: str, outdir: Path, fname: str, what: str):
    plt.figure(figsize=(7, 4))
    vals = [a_only, both, b_only]
    labels = [f"Exclusivo A (Qwen)\n{a_only}", f"Interseção\n{both}", f"Exclusivo B (Llama)\n{b_only}"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    # Gráfico 'equivalente ao Venn' com barras empilhadas (categoria única)
    ax = plt.gca()
    bottom = 0
    for v, lab, color in zip(vals, labels, colors):
        ax.bar(["Sobreposição"], [v], bottom=bottom, label=lab, color=color)
        bottom += v
    plt.title(f"Sobreposição de {what} - {prompt}\n{MODEL_A_LABEL} vs {MODEL_B_LABEL}")
    plt.xlabel("Categoria")
    plt.ylabel("Contagem")
    plt.legend(title="Partição")
    style_axes(plt.gca())
    add_footnote(plt.gcf(), f"Números de A-only, Interseção e B-only para {what} no {prompt}.")
    plt.tight_layout()
    out_path = outdir / fname
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_coverage(coverage_df: pd.DataFrame, outdir: Path, fname: str):
    if coverage_df is None or len(coverage_df) == 0 or "coverage" not in coverage_df:
        print("[WARN] Sem dados de cobertura para plotar; pulando.")
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=coverage_df, x="prompt", y="coverage", hue="modelo")
    plt.title(f"Cobertura por Prompt (P1–P3)\n{MODEL_A_LABEL} vs {MODEL_B_LABEL}")
    plt.xlabel("Prompt")
    plt.ylabel("Cobertura (|Modelo| / |União|)")
    plt.ylim(0, 1.05)
    plt.legend(title="Modelos")
    style_axes(plt.gca())
    add_footnote(plt.gcf(), "Cobertura: fração do universo combinado de entidades (A∪B) coberta por cada modelo.")
    plt.tight_layout()
    out_path = outdir / fname
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hallucination(halluc_df: pd.DataFrame, outdir: Path, fname: str):
    if halluc_df is None or len(halluc_df) == 0 or "hallucination_rate" not in halluc_df:
        print("[WARN] Sem dados de alucinação para plotar; pulando.")
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=halluc_df, x="prompt", y="hallucination_rate", hue="modelo")
    plt.title(f"Taxa de Alucinação (estimada) por Prompt (P1–P3)\n{MODEL_A_LABEL} vs {MODEL_B_LABEL}")
    plt.xlabel("Prompt")
    plt.ylabel("Proporção de itens não encontrados")
    plt.ylim(0, 1.05)
    plt.legend(title="Modelos")
    style_axes(plt.gca())
    add_footnote(plt.gcf(), "Heurística: verifica termos-chave do item no texto do artigo (se fornecido) ou nas citações + raw_text do próprio JSON.")
    plt.tight_layout()
    out_path = outdir / fname
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_f1_heatmap(f1_entities: Dict[str, float], f1_relations: Dict[str, float], outdir: Path):
    df_e = pd.DataFrame({"prompt": ["P1", "P2", "P3"], "F1": [f1_entities.get(p, np.nan) for p in ["P1", "P2", "P3"]]})
    df_r = pd.DataFrame({"prompt": ["P1", "P2", "P3"], "F1": [f1_relations.get(p, np.nan) for p in ["P1", "P2", "P3"]]})

    for title, df, fname in [
        ("F1-score (Entidades) P1–P3", df_e, "heatmap_f1_entities_P1_P3.png"),
        ("F1-score (Relações) P1–P3", df_r, "heatmap_f1_relations_P1_P3.png"),
    ]:
        plt.figure(figsize=(6, 2.5))
        pivot = df.set_index("prompt").T
        sns.heatmap(pivot, annot=True, cmap="Greens", vmin=0, vmax=1, cbar=True, fmt=".2f")
        plt.title(f"{title}\n{MODEL_A_LABEL} vs {MODEL_B_LABEL}")
        plt.yticks(rotation=0)
        add_footnote(plt.gcf(), "F1 simétrico = 2|A∩B|/(|A|+|B|).")
        plt.tight_layout()
        out_path = outdir / fname
        plt.savefig(out_path, dpi=200)
        plt.close()

def entity_sets_from_agg(entities_by_type):
    typed = set()
    untyped = set()
    for etype, s in entities_by_type.items():
        for e in s:
            typed.add((e, etype))   # identidade com tipo (igual às tabelas)
            untyped.add(e)          # identidade só por texto (se precisar no futuro)
    return typed, untyped

# ----------------------------
# Relatório
# ----------------------------
def write_summary(
    outdir: Path,
    jacc_entities: Dict[str, float],
    jacc_relations: Dict[str, float],
    f1_entities: Dict[str, float],
    f1_relations: Dict[str, float],
    coverage_rows: List[Dict[str, Any]],
    halluc_rows: List[Dict[str, Any]],
    grouped_jaccard_by_type: Dict[str, Dict[str, float]],
):
    path = outdir / "summary.md"
    lines = []
    lines.append("# Resumo da Análise")
    lines.append("")
    lines.append(f"- Modelos comparados: {MODEL_A_LABEL}; {MODEL_B_LABEL}")
    lines.append("- Similaridades: Jaccard (entidades e relações) por P1–P8; F1-simétrico por P1–P3.")
    lines.append("- Cobertura: fração do universo combinado de entidades (P1–P3) coberta por cada modelo.")
    lines.append("- Taxa de alucinação (estimada): fração de itens cujos termos-chave não aparecem no texto base.")
    lines.append("")
    lines.append("## Jaccard por Prompt (Entidades)")
    for p in PROMPTS:
        val = jacc_entities.get(p, np.nan)
        lines.append(f"- {p} ({PROMPT_DESCRIPTIONS.get(p,'')}): {val:.3f}" if not np.isnan(val) else f"- {p} ({PROMPT_DESCRIPTIONS.get(p,'')}): NaN")
    lines.append("")
    lines.append("## Jaccard por Prompt (Relações)")
    for p in PROMPTS:
        val = jacc_relations.get(p, np.nan)
        lines.append(f"- {p} ({PROMPT_DESCRIPTIONS.get(p,'')}): {val:.3f}" if not np.isnan(val) else f"- {p} ({PROMPT_DESCRIPTIONS.get(p,'')}): NaN")
    lines.append("")
    lines.append("## Jaccard por Tipo de Entidade (P1–P8)")
    for p, d in grouped_jaccard_by_type.items():
        lines.append(f"- {p}:")
        for et, val in d.items():
            lines.append(f"  - {et}: {val:.3f}" if not np.isnan(val) else f"  - {et}: NaN")
    lines.append("")
    lines.append("## F1-simétrico (P1–P3)")
    for p in ["P1", "P2", "P3"]:
        fe = f1_entities.get(p, np.nan)
        fr = f1_relations.get(p, np.nan)
        lines.append(f"- {p} Entidades F1: {fe:.3f} | Relações F1: {fr:.3f}" if not (np.isnan(fe) and np.isnan(fr)) else f"- {p} Entidades F1: NaN | Relações F1: NaN")
    lines.append("")
    lines.append("## Cobertura (P1–P3)")
    for row in coverage_rows:
        lines.append(f"- {row['prompt']} {row['modelo']}: {row['coverage']:.3f}" if not np.isnan(row["coverage"]) else f"- {row['prompt']} {row['modelo']}: NaN")
    lines.append("")
    lines.append("## Taxa de Alucinação (estimada, P1–P3)")
    for row in halluc_rows:
        rate = row["hallucination_rate"]
        val = f"{rate:.3f}" if not np.isnan(rate) else "NaN"
        lines.append(f"- {row['prompt']} {row['modelo']}: {val} (n={row['n_items']} itens)")
    lines.append("")
    lines.append("## Notas metodológicas")
    lines.append("- F1-score é simétrico (não requer referência externa).")
    lines.append("- Cobertura usa o universo combinado (A∪B) por prompt, medindo |A|/|A∪B| e |B|/|A∪B|.")
    lines.append("- 'Alucinação' é heurística baseada na presença de termos do item no texto do artigo (se fornecido) ou nas citações + raw_text do próprio JSON.")
    lines.append("- Relações padronizadas por prompt: ex., em P2 usamos alvo 'lignin' para comparabilidade.")
    lines.append("- Quando ambos os modelos não retornam itens em um prompt, Jaccard/F1 podem ser NaN.")
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Pipeline principal
# ----------------------------
def main(args):
    outdir = Path(args.outdir)
    figs_dir = outdir / "figures"
    tables_dir = outdir / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Carrega
    model_a_records = load_jsonl(Path(args.model_a))
    model_b_records = load_jsonl(Path(args.model_b))
    article_texts = load_article_texts(Path(args.article_texts) if args.article_texts else None)

    # Agrega por prompt
    aggA = aggregate_by_prompt(model_a_records)
    aggB = aggregate_by_prompt(model_b_records)

    # Jaccard por prompt (entidades e relações)
    jacc_entities = {}
    jacc_relations = {}
    grouped_jaccard_by_type = {}

    # P1–P3: contagens, F1, cobertura, alucinação e tabelas
    entity_counts_A = {}
    entity_counts_B = {}
    f1_entities = {}
    f1_relations = {}
    coverage_rows = []
    halluc_rows = []

    for p in PROMPTS:
        entsA_by_type = aggA[p]["entities_by_type"]
        entsB_by_type = aggB[p]["entities_by_type"]
        setA_typed, setA_untyped = entity_sets_from_agg(entsA_by_type)
        setB_typed, setB_untyped = entity_sets_from_agg(entsB_by_type)
        setA_e_all = setA_typed
        setB_e_all = setB_typed


        jacc_entities[p] = jaccard(setA_e_all, setB_e_all)

        setA_r = aggA[p]["relations"]
        setB_r = aggB[p]["relations"]
        jacc_relations[p] = jaccard(setA_r, setB_r)

        # Jaccard por tipo
        types_all = set(list(entsA_by_type.keys()) + list(entsB_by_type.keys()))
        grouped_jaccard_by_type[p] = {}
        for et in sorted(types_all):
            grouped_jaccard_by_type[p][et] = jaccard(entsA_by_type.get(et, set()), entsB_by_type.get(et, set()))

        if p in ["P1", "P2", "P3"]:
            entity_counts_A[p] = len(setA_e_all)
            entity_counts_B[p] = len(setB_e_all)

            f1_entities[p] = f1_symmetric(setA_e_all, setB_e_all)
            f1_relations[p] = f1_symmetric(setA_r, setB_r)

            covA, covB = coverage(setA_e_all, setB_e_all)
            coverage_rows.append({"prompt": p, "modelo": MODEL_A_LABEL, "coverage": covA})
            coverage_rows.append({"prompt": p, "modelo": MODEL_B_LABEL, "coverage": covB})

            hallA = compute_hallucination_rate(aggA[p], article_texts, p)
            hallB = compute_hallucination_rate(aggB[p], article_texts, p)
            halluc_rows.append({"prompt": p, "modelo": MODEL_A_LABEL, **hallA})
            halluc_rows.append({"prompt": p, "modelo": MODEL_B_LABEL, **hallB})

            # Tabelas comparativas (entidades e relações)
            prepare_tables_for_prompt(p, aggA[p], aggB[p], tables_dir)

    # Salvar métricas
    pd.DataFrame([
        {"prompt": p, "jaccard_entities": jacc_entities[p], "jaccard_relations": jacc_relations[p]}
        for p in PROMPTS
    ]).to_csv(outdir / "metrics_jaccard_by_prompt.csv", index=False)

    rows_type = []
    for p, d in grouped_jaccard_by_type.items():
        for et, val in d.items():
            rows_type.append({"prompt": p, "entity_type": et, "jaccard": val})
    pd.DataFrame(rows_type).to_csv(outdir / "metrics_jaccard_by_type.csv", index=False)

    pd.DataFrame([
        {"prompt": p, "f1_entities": f1_entities.get(p, np.nan), "f1_relations": f1_relations.get(p, np.nan)}
        for p in ["P1", "P2", "P3"]
    ]).to_csv(outdir / "metrics_f1_P1_P3.csv", index=False)

    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(outdir / "metrics_coverage_P1_P3.csv", index=False)

    hall_df = pd.DataFrame(halluc_rows)
    # Garantir nome esperado da coluna
    if "rate" in hall_df.columns and "hallucination_rate" not in hall_df.columns:
        hall_df = hall_df.rename(columns={"rate": "hallucination_rate"})
    hall_df.to_csv(outdir / "metrics_hallucination_P1_P3.csv", index=False)

    # Gráficos
    plot_jaccard_heatmaps(jacc_entities, jacc_relations, figs_dir)
    plot_entity_counts_bar(entity_counts_A, entity_counts_B, ["P1", "P2", "P3"], figs_dir, "bar_entity_counts_P1_P3.png")
    plot_f1_heatmap(f1_entities, f1_relations, figs_dir)
    plot_coverage(cov_df, figs_dir, "bar_coverage_P1_P3.png")
    plot_hallucination(hall_df, figs_dir, "bar_hallucination_P1_P3.png")

    # Gráficos de sobreposição (equivalente a Venn) para P1–P3
    for p in ["P1", "P2", "P3"]:
        entsA_by_type = aggA[p]["entities_by_type"]
        entsB_by_type = aggB[p]["entities_by_type"]
        setA_typed, _ = entity_sets_from_agg(entsA_by_type)
        setB_typed, _ = entity_sets_from_agg(entsB_by_type)
        a_only_e = len(setA_typed - setB_typed)
        both_e   = len(setA_typed & setB_typed)
        b_only_e = len(setB_typed - setA_typed)
        plot_overlap_equivalent(a_only_e, both_e, b_only_e, p, figs_dir, f"overlap_entities_{p}.png", "entidades")

        setA_r = aggA[p]["relations"]
        setB_r = aggB[p]["relations"]
        a_only_r = len(setA_r - setB_r)
        both_r = len(setA_r & setB_r)
        b_only_r = len(setB_r - setA_r)
        plot_overlap_equivalent(a_only_r, both_r, b_only_r, p, figs_dir, f"overlap_relations_{p}.png", "relações")

    # Resumo
    write_summary(
        outdir=outdir,
        jacc_entities=jacc_entities,
        jacc_relations=jacc_relations,
        f1_entities=f1_entities,
        f1_relations=f1_relations,
        coverage_rows=coverage_rows,
        halluc_rows=halluc_rows,
        grouped_jaccard_by_type=grouped_jaccard_by_type,
    )

    print(f"Concluído. Resultados em: {outdir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparação Qwen vs Llama em JSONL por prompts P1–P8 (formato rebuild_jsonl_with_rawtext.py).")
    parser.add_argument("--model_a", required=True, help="Caminho para o JSONL do Modelo A (Qwen)")
    parser.add_argument("--model_b", required=True, help="Caminho para o JSONL do Modelo B (Llama)")
    parser.add_argument("--outdir", default="results", help="Diretório de saída (default: results)")
    parser.add_argument("--article_texts", default=None, help="Opcional: JSON com textos dos artigos (pmid -> texto) para verificar 'alucinação'")
    args = parser.parse_args()
    main(args)