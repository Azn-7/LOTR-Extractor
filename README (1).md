# LOTR Extractor

A knowledge graph extraction pipeline for the Lord of the Rings book series. Reads raw LOTR text files, identifies named entities using GLiNER, deduplicates them, and produces structured node/edge data suitable for graph visualization or analysis.

## Pipeline Overview

1. **Text ingestion** — Reads `.txt` files from a directory, splits by chapter boundaries
2. **Sliding window** — 50-word windows with a 5-word stride fed into a batch queue
3. **GLiNER NER** — Batch inference extracts entities (PERSON, ORGANIZATION, LOCATION, EVENT)
4. **Node deduplication** — Entities normalized to lowercase keys; reference counts tracked
5. **Alias resolution** — Variant names map to a canonical full name (e.g. `"bilbo"` → `"Bilbo Baggins"`)
6. **Exception filtering** — Pronouns, generic roles, and artifacts are discarded
7. **Output** — `nodes.txt` (CSV-formatted), `edges.txt` (planned)

## Requirements

- Python 3.x
- `gliner`
- `spacy`
- `regex`
- `torch`

Install dependencies:
```bash
pip install gliner spacy regex torch
```

## Usage

Place your LOTR `.txt` files in the `LOTR_Text/` directory, then run:

```bash
python LOTR_Extractor.py
```

Output is written to `nodes.txt` in CSV format, readable directly by pandas:

```python
import pandas as pd
df = pd.read_csv("nodes.txt")
```

## File Structure

```
LOTR_Extractor.py     # Main pipeline
Extractor_Utils.py    # Utility functions (edge helpers etc.)
nodes.txt             # Extracted node data (CSV)
edges.txt             # Extracted edge data (planned)
LOTR_Text/            # Raw input .txt files
```

## Output Format

`nodes.txt` columns:

| Column | Description |
|---|---|
| Node_type | PERSON, ORGANIZATION, LOCATION, or EVENT |
| ID | Unique identifier (e.g. `PER_BILBO_BAGGINS`) |
| Description | Entity description (currently `[N/A]`) |
| Label | Canonical display name |
| Reference_Count | Number of times entity appeared |
| First_Apperance | Book and chapter of first appearance |
| Last_Apperance | Book and chapter of last appearance |
| Confidence | GLiNER confidence score (0–1) |

## Roadmap

- [ ] spaCy co-reference resolution (`"the old hobbit"` → `Bilbo Baggins`)
- [ ] Edge extraction (co-occurrence within a window = relationship)
- [ ] pandas integration for output
- [ ] Programmatic alias detection via fuzzy matching on `nodes.txt`
