# TCC

Este repositório foi criado para armazenar os *scripts* desenvolvidos no meu TCC, cujo objetivo foi realizar extração de entidades e relações em artigos científicos relacionados ao metabolismo da lignina usando diferentes modelos de IA (Qwen, Llama, GPT e Gemini).

O foco deste README é facilitar a reprodução dos resultados, permitindo que outros usuários utilizem ou adaptem os códigos futuramente.

1. [store_pubmed_articles.py](https://github.com/DaniTeix/TCC/blob/fd232dabf0fbef4663b69c0af630153f39dc74ba/store_pubmed_articles.py)

Este *script* é responsável por baixar artigos de acesso aberto e gratuito da base PubMed, utilizando uma lista de PMIDs.

Funcionamento: 
* Recebe uma lista de PMIDs (no meu uso original, a lista estava no próprio código).
* Faz requisições à API do PubMed/Entrez e salva os artigos em formato XML/JSON.
* Atualmente, o e-mail do usuário (necessário para o Entrez) é colocado diretamente no código — mas essa parte pode ser facilmente adaptada para receber um argumento via terminal.

Bibliotecas utilizadas:
```bash
import requests
import json
import time
from xml.etree import ElementTree as ET
from Bio import Entrez
```

Como executar: 
```bash
python3 store_pubmed_articles.py
```

Observações: Recomenda-se adaptar o script para aceitar argumentos, como por exemplo:
-pmid_file lista_pmids.txt
-email seu_email
Isso torna a execução mais flexível e reprodutível.

2. [extraction_qwen_or_llama.groq.py](https://github.com/DaniTeix/TCC/blob/fd232dabf0fbef4663b69c0af630153f39dc74ba/extraction_qwen_or_llama.groq.py)

Este *script* acessa os modelos *Qwen* ou *Llama* usando a plataforma Groq, aplicando extração de relações em lotes de artigos previamente baixados.

Requisito importante: chave de acesso Groq
* Antes de executar, gere sua chave na plataforma da Groq e exporte no terminal:
```bash
export GROQ_API_KEY='sua-chave'
```

Bibliotecas necessárias: 
```bash
import os
import re
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple
```
Como executar: 
```bash
python3 extraction_qwen_or_llama.groq.py \
  --input 'arquivo_com_artigos.json' \
  --output 'saida.jsonl' \
  --model 'qwen/qwen3-32b'  # ou 'llama-3.1-8b-instant'
```
Observações:
* O script foi pensado para ser flexível quanto ao modelo usado.
* A entrada deve conter artigos já processados pelo script anterior.
* A saída é um arquivo .jsonl com as extrações geradas.

3. [compare_models.py](https://github.com/DaniTeix/TCC/blob/fd232dabf0fbef4663b69c0af630153f39dc74ba/compare_models.py)

Este script realiza a comparação entre as respostas de dois modelos distintos, utilizando os arquivos .jsonl gerados anteriormente.
Ele produz tabelas e gráficos com estatísticas relevantes.

Bibliotecas utilizadas: 
```bash
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Como executar:
```bash
python3 compare_models.py \
  --model_a modeloA.jsonl \
  --model_b modeloB.jsonl \
  --outdir resultados/
```
Argumentos obrigatórios:
* model_a
* model_b

## Considerações finais
* Todos os scripts podem ser facilmente adaptados para aceitar argumentos adicionais.
* *Recomenda-se criar um ambiente virtual para rodar as dependências.
