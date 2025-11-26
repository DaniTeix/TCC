#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversor de JSON (lista ou dicionário de artigos) para o formato esperado pelo compare_models.py

Entrada:
  - JSON com registros que tenham os campos: pmid, title, abstract, full_text

Saída:
  - JSON no formato { pmid: "title abstract full_text" }

Uso:
  python3 convert_json_for_json.py --input articles_with_fulltext5.json --output articles_for_hallucination.json
"""

import json
import argparse
import sys
import os

def converter_json_para_mapeamento(input_path: str, output_path: str) -> None:
    """Converte um JSON comum (lista ou dict) no formato pmid -> texto completo."""
    if not os.path.exists(input_path):
        print(f"❌ Arquivo de entrada não encontrado: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}

    # Caso 1: lista de artigos
    if isinstance(data, list):
        for idx, rec in enumerate(data, start=1):
            pmid = str(rec.get("pmid", "")).strip()
            if not pmid:
                print(f"⚠️ Registro {idx} sem PMID — ignorado.")
                continue

            text_parts = [
                str(rec.get("title", "")).strip(),
                str(rec.get("abstract", "")).strip(),
                str(rec.get("full_text", "")).strip()
            ]
            text = " ".join([t for t in text_parts if t])
            if text:
                out[pmid] = text

    # Caso 2: dicionário com PMIDs como chaves
    elif isinstance(data, dict):
        for pmid, rec in data.items():
            pmid = str(pmid).strip()
            if not isinstance(rec, dict):
                print(f"⚠️ Valor inesperado para {pmid}: esperado dict, obtido {type(rec).__name__}")
                continue

            text_parts = [
                str(rec.get("title", "")).strip(),
                str(rec.get("abstract", "")).strip(),
                str(rec.get("full_text", "")).strip()
            ]
            text = " ".join([t for t in text_parts if t])
            if text:
                out[pmid] = text
    else:
        print("❌ Estrutura de JSON não reconhecida: deve ser lista ou dicionário.")
        sys.exit(1)

    with open(output_path, "w", encoding="utf-8") as g:
        json.dump(out, g, ensure_ascii=False, indent=2)

    print(f"✅ Gerado: {output_path} com {len(out)} entradas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converter JSON (artigos) para JSON no formato pmid->texto")
    parser.add_argument("--input", "-i", required=True, help="Arquivo de entrada (.json)")
    parser.add_argument("--output", "-o", default="article_texts.json", help="Arquivo de saída (.json)")
    args = parser.parse_args()

    converter_json_para_mapeamento(args.input, args.output)
