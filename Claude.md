# LOTR Extractor — Project Context

## Token Safety
Before performing any task that would consume a significant number of tokens (e.g. reading large files, loading long outputs into context, bulk processing), Claude must first warn the user and explain why it would be expensive, then wait for explicit confirmation before proceeding. If in doubt, do not proceed.

## Project-Goal
Build a knowledge graph extraction pipeline for the Lord of the Rings book series.
The pipeline reads raw LOTR text files, identifies named entities, deduplicates them,
and produces structured node/edge data suitable for graph visualization or analysis.

## Pipeline Overview

1. **Text ingestion** — Reads `.txt` files from a directory, splits by chapter boundaries
2. **Sliding window** — 50-word windows with a 5-word stride fed into a batch queue
3. **GLiNER NER** — Batch inference extracts entities (PERSON, ORGANIZATION, LOCATION, EVENT)
4. **Node deduplication** — Entities are normalized to lowercase keys; reference counts tracked
5. **Alias resolution** — Short/variant names map to a canonical full name (e.g. "bilbo" → "Bilbo Baggins")
6. **Exception filtering** — Pronouns, generic roles, and artifacts are discarded
7. **Output** — `nodes.txt` (CSV-formatted), `edges.txt` (planned)

## Current Status

### Done
- GLiNER batch inference pipeline working
- Sliding window reader with chapter-reset
- Node deduplication with lowercase key normalization
- Exceptions list (pronouns, generic roles, artifacts, misclassified entities)
- Alias dictionary pre-populated for Book 1 Chapter 1 characters
- nodes.txt output in CSV format (pandas-ready)

### In Progress
- Alias resolution wiring: when a GLiNER entity matches an alias key, the update/insert
  should target the canonical entity's key in node_data, not create a new entry
- First/Last appearance tracking is currently [N/A] — needs window index or absolute
  character offset, not the within-window char position GLiNER returns

### Not Started
- spaCy co-reference resolution (will help deduplicate "the old hobbit" → Bilbo Baggins)
- Edge extraction (co-occurrence within a window = relationship between entities)
- pandas integration (trivial once data is clean — swap txt write for DataFrame.to_csv)

## Key Design Decisions

- **Nodes before edges** — edges require clean entity data; noisy nodes = garbage connections
- **GLiNER first, spaCy later** — spaCy co-reference is the biggest quality jump but requires
  more setup; GLiNER alone handles the bulk of named entity detection
- **Alias dict is manual** — programmatic alias detection (fuzzy match on nodes.txt) is planned
  but done by eye for now; nodes.txt is the right place to audit, not the raw book text
- **node_data keyed by canonical lowercase name** — alias variants must resolve to this key
  before any insert or update

## File Structure

- `LOTR_Extractor.py` — main pipeline
- `Extractor_Utils.py` — utility functions (edge helpers etc.)
- `nodes.txt` — extracted node data (CSV, one entity per line)
- `edges.txt` — extracted edge data (planned)
- `LOTR_Text/` — raw input text files

## Known Issues / Watch-outs

- `alias` dict values are title-cased ("Bilbo Baggins") but node_data keys are lowercase —
  always call `.lower()` when using an alias value as a lookup key
- GLiNER `start`/`end` are character offsets within the window string, not the full book
- "hobbit" and hobbit-variants are intentionally NOT in exceptions — they may alias to a
  specific character once co-reference is resolved
- "took" and "proudfoot" in alias are risky — both are family names used for multiple characters
