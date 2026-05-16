from gliner import GLiNER
import regex
from collections import deque
from pathlib import Path
import itertools
import pandas as pd
import Extractor_Utils
import requests
from tqdm import tqdm

try:
    from dedup_utils import deduplicate_entities
except ImportError:
    def deduplicate_entities(nd):
        return nd

# ==============================================================================
# =============================== CONFIGURATION ================================
# ==============================================================================

directory_LOTR = Path(r"C:\Coding\LOTR-Extractor\LOTR_Text")
TEST_MODE = True         # Set to False to process all LOTR_Text files
GENERATE_DESCRIPTIONS = False  # Set to False to skip Ollama and write [N/A] descriptions

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL   = "http://localhost:11434/api/chat"

# ==============================================================================
# =============================== INITALIZATION ================================
# ==============================================================================

# Data Variables
node_data = {}  # {canonical_lowercase_key: attribute_dict}
edge_data = {}  # {frozenset({src, tgt}): edge_dict}
added_entities = set()

# Patterns
chapter_pattern = regex.compile(r"_Chapter\s(\d+)_")   # captures chapter number

# Appearance Tracker
book_vol = 1
chapter = 0

# Deque (Read 50 words each, advance 5 words at a time)
WINDOW_SIZE = 50
STRIDE = 5

# GLiNER
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1").to('cuda')
labels = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT"]
batch_queue = []       # list of window strings queued for batch inference
window_meta = []       # parallel list of (book_vol, chapter) per window
BATCH_SIZE = 64

# Alias Dictionary
alias = {
    # Alias : Main name (key is lowercase alias, value is canonical full name)
    "bilbo": "Bilbo Baggins",
    "mr. bilbo": "Bilbo Baggins",
    "mr. bilbo baggins": "Bilbo Baggins",
    "uncle bilbo": "Bilbo Baggins",
    "mr. baggins": "Bilbo Baggins",
    "frodo": "Frodo Baggins",
    "mr. frodo": "Frodo Baggins",
    "poor mr. frodo": "Frodo Baggins",
    "gandalf": "Gandalf the Grey",
    "gandalf the wizard": "Gandalf the Grey",
    "the wizard": "Gandalf the Grey",
    "old wizard": "Gandalf the Grey",
    "gaffer": "Old Gaffer Gamgee",
    "the gaffer": "Old Gaffer Gamgee",
    "old ham gamgee": "Old Gaffer Gamgee",
    "master hamfast": "Old Gaffer Gamgee",
    "sam": "Sam Gamgee",
    "knowledgeable sam": "Sam Gamgee",
    "merry": "Merry Brandybuck",
    "old rory": "Rory Brandybuck",
    "old rory brandybuck": "Rory Brandybuck",
    "rory": "Rory Brandybuck",
    "otho": "Otho Sackville-Baggins",
    "lobelia": "Lobelia Sackville-Baggins",
    "drogo": "Drogo Baggins",
    "mr. drogo": "Drogo Baggins",
    "mr. drogo baggins": "Drogo Baggins",
    "miss primula": "Miss Primula Brandybuck",
    "old took": "Old Took",
    "took": "Old Took",
    "odo": "Odo Proudfoot",
    "old odo proudfoot": "Odo Proudfoot",
    "sancho": "Sancho Proudfoot",
    "young sancho proudfoot": "Sancho Proudfoot",
    "proudfoot": "Odo Proudfoot",
    # Additional aliases found in output
    "mr. gandalf": "Gandalf the Grey",
    "old gandalf": "Gandalf the Grey",
    "old bilbo": "Bilbo Baggins",
    "old mr. bilbo": "Bilbo Baggins",
    "mr. merry": "Merry Brandybuck",
    "old dad": "Old Gaffer Gamgee",
}

exceptions = [
    # Pronouns
    "he", "him", "they", "she", "her", "i", "you", "me", "my", "his", "your", "it", "were",
    "we", "us", "our", "them", "their", "theirs",
    "he escorted her",
    # Generic roles
    "local legend", "miller", "the miller", "friend", "stranger", "the stranger",
    "visitor", "host", "their host", "person", "somebody", "young fellow",
    "old man", "an old man", "the old man", "people", "some people", "other people",
    "all and sundry", "his guests", "everyone", "cooks", "postman",
    "voluntary assistant postmen", "knowledgeable", "old", "gross",
    # Generic nouns misclassified as entities
    "deal", "sale", "plot", "evil", "luck", "fate", "fear", "loss", "race", "find",
    "hero", "vote", "aid", "body", "work", "joy", "bane", "raid", "spy", "rear",
    "dusk", "haze", "fort", "grip", "loon", "hiss", "bath", "log", "logs", "ship",
    "hut", "gore", "slot", "dogs", "rams", "heed", "dais", "slab", "foam",
    # Artifacts / incomplete
    "mr.", "announcement_", "morning_",
    # Misclassified
    "bucklanders", "shire-folk", "chief table", "boats",
]

# ==============================================================================
# ============================= EDGE HELPERS ===================================
# ==============================================================================

def _upsert_edge(src, tgt, context, source_document, chapter, book_vol):
    """Insert a new edge or increment weight on an existing one."""
    key = frozenset({src, tgt})
    if key in edge_data:
        edge_data[key]["Weight"] += 1
        edge_data[key]["Context"] = context
    else:
        src_type = node_data.get(src, {}).get('Node_type', 'UNKNOWN')
        tgt_type = node_data.get(tgt, {}).get('Node_type', 'UNKNOWN')
        label = f"{src_type}-{tgt_type}"
        edge_data[key] = Extractor_Utils.add_edge(
            source=src,
            target=tgt,
            label=label,
            type="CO_OCCURRENCE",
            context=context,
            source_document=source_document,
            weight=1,
        )
        edge_data[key]["Chapter"] = chapter
        edge_data[key]["Book_Vol"] = book_vol
        edge_data[key]["Description"] = "[N/A]"


# ==============================================================================
# ============================== MAIN FUNCTIONS ================================
# ==============================================================================

def execute_GLiNER(batch_queue, window_meta_list, book_vol, chapter):
    global edge_data, added_entities

    print(f"    Running GLiNER on {len(batch_queue)} windows...", end="", flush=True)
    batch_results = model.inference(batch_queue, labels, threshold=0.5, batch_size=len(batch_queue))
    print(f" done  (entities so far: {len(added_entities)})")

    for window_idx, window_entities in enumerate(batch_results):

        window_text = batch_queue[window_idx]
        w_book_vol, w_chapter = window_meta_list[window_idx]

        window_canonical_names = []

        for entity_dict in window_entities:
            raw_text = entity_dict["text"].strip("_").strip()

            # Drop sentence fragments and overly long noun phrases
            if len(raw_text) > 40:
                continue

            if entity_dict.get("_coref_resolved"):
                name = raw_text
                canonical_display = name.title()
            else:
                name_lower = raw_text.lower()
                canonical_value = alias.get(name_lower, None)
                if canonical_value is not None:
                    canonical_display = canonical_value
                    name = canonical_value.lower()
                else:
                    name = name_lower
                    canonical_display = raw_text.title()

            if name in exceptions:
                continue

            # Update existing entity
            if name in added_entities:
                node_data[name]['Last_Apperance'] = f"Book {w_book_vol}, Chapter {w_chapter}"
                node_data[name]['Reference_Count'] += 1
                window_canonical_names.append(name)
                continue

            # Create new entity
            attribute = {}
            attribute['Node_type'] = entity_dict['label']
            if entity_dict['label'] == "EVENT":
                attribute['ID'] = ("EVT" + " " + name).replace(" ", "_").upper()
            else:
                attribute['ID'] = (entity_dict['label'][:3] + " " + name).replace(" ", "_").upper()
            attribute['Description'] = '[N/A]'
            attribute['Label'] = canonical_display
            attribute['Reference_Count'] = 1
            attribute['First_Apperance'] = f"Book {w_book_vol}, Chapter {w_chapter}"
            attribute['Last_Apperance'] = f"Book {w_book_vol}, Chapter {w_chapter}"
            attribute['Confidence'] = round(entity_dict['score'], 2)
            attribute['_context'] = window_text

            added_entities.add(name)
            node_data[name] = attribute
            window_canonical_names.append(name)

        # Edge extraction
        unique_entities = sorted(set(window_canonical_names))
        if len(unique_entities) >= 2:
            for src, tgt in itertools.combinations(unique_entities, 2):
                _upsert_edge(
                    src=src,
                    tgt=tgt,
                    context=window_text,
                    source_document=str(text_file_path),
                    chapter=w_chapter,
                    book_vol=w_book_vol,
                )

    batch_queue.clear()
    window_meta_list.clear()
    return


def generate_descriptions(node_data: dict) -> dict:
    entities = [(key, attrs) for key, attrs in node_data.items() if attrs.get('_context')]

    if not entities:
        for attrs in node_data.values():
            attrs.pop('_context', None)
        return node_data

    DESC_BATCH = 20
    system_prompt = (
        "You are a literary analyst for Lord of the Rings. "
        "For each entity, write a single concise sentence describing who or what it is "
        "based on the provided context window. Focus on the entity's role, relationships, "
        "and nature — not plot events. Use the format:\n"
        "entity_key: One-line description.\n\n"
        "Example:\n"
        "bilbo baggins: Bilbo Baggins is a well-to-do hobbit who lives at Bag End in the Shire."
    )

    for i in tqdm(range(0, len(entities), DESC_BATCH), desc="  Describing entities", unit="batch"):
        batch = entities[i : i + DESC_BATCH]
        lines = []
        for key, attrs in batch:
            lines.append(
                f"Name: {key}\n"
                f"Type: {attrs.get('Node_type', 'ENTITY')}\n"
                f"Context: {attrs['_context']}"
            )
        user_msg = "Generate descriptions for these entities:\n\n" + "\n\n---\n\n".join(lines)

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        response_text = response.json()["message"]["content"]

        for line in response_text.strip().splitlines():
            if ':' not in line:
                continue
            key_part, desc_part = line.split(':', 1)
            canonical = key_part.strip().lower()
            if canonical in node_data:
                node_data[canonical]['Description'] = desc_part.strip()

    for attrs in node_data.values():
        attrs.pop('_context', None)

    return node_data


def generate_edge_descriptions(edge_data: dict) -> dict:
    edges = [(k, v) for k, v in edge_data.items() if v.get("Weight", 1) >= 2]

    if not edges:
        return edge_data

    DESC_BATCH = 20
    system_prompt = (
        "You are a literary analyst for Lord of the Rings. "
        "For each pair of entities, write a single concise sentence describing their relationship "
        "based on the provided context window. Focus on how the two entities are connected — "
        "e.g. kinship, alliance, conflict, location, ownership. Use the format:\n"
        "EDGE_INDEX: One-line relationship description.\n\n"
        "Example:\n"
        "0: Bilbo Baggins is the uncle and guardian of Frodo Baggins."
    )

    for i in tqdm(range(0, len(edges), DESC_BATCH), desc="  Describing edges", unit="batch"):
        batch = edges[i : i + DESC_BATCH]
        lines = []
        for idx, (key, attrs) in enumerate(batch):
            src_label = node_data.get(attrs["Source"], {}).get("Label", attrs["Source"])
            tgt_label = node_data.get(attrs["Target"], {}).get("Label", attrs["Target"])
            src_type = node_data.get(attrs["Source"], {}).get("Node_type", "ENTITY")
            tgt_type = node_data.get(attrs["Target"], {}).get("Node_type", "ENTITY")
            lines.append(
                f"Index: {idx}\n"
                f"Source: {src_label} (Type: {src_type})\n"
                f"Target: {tgt_label} (Type: {tgt_type})\n"
                f"Context: {attrs.get('Context', '')}"
            )
        user_msg = "Generate relationship descriptions for these entity pairs:\n\n" + "\n\n---\n\n".join(lines)

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        response_text = response.json()["message"]["content"]

        for line in response_text.strip().splitlines():
            if ':' not in line:
                continue
            idx_part, desc_part = line.split(':', 1)
            try:
                idx = int(idx_part.strip())
            except ValueError:
                continue
            if 0 <= idx < len(batch):
                key = batch[idx][0]
                edge_data[key]["Description"] = desc_part.strip()

    return edge_data


def LOTR_Extractor():
    global chapter, book_vol, text_file_path

    window = deque(maxlen=WINDOW_SIZE)

    text_files = [Path("test.txt")] if TEST_MODE else sorted(directory_LOTR.glob("*.txt"))

    for text_file_path in text_files:
        print(f"\n============ READING: {text_file_path.name} ============")
        word_counter = 0
        window.clear()

        with open(text_file_path, 'r', encoding='cp1252') as file:
            for line in file:
                match = chapter_pattern.search(line)
                if match:
                    new_chapter = int(match.group(1))
                    # Chapter number reset means a new book has started
                    if new_chapter <= chapter:
                        book_vol += 1
                        print(f"  -> Book {book_vol} detected")
                    chapter = new_chapter
                    window.clear()
                    word_counter = 0
                    continue

                words = line.split()
                for word in words:
                    window.append(word)
                    word_counter += 1
                    if len(window) == WINDOW_SIZE:
                        if word_counter % STRIDE == 0 and chapter > 0:
                            gliner_window = " ".join(window)
                            batch_queue.append(gliner_window)
                            window_meta.append((book_vol, chapter))

                if len(batch_queue) >= BATCH_SIZE:
                    print(f"  Batch ready (size {len(batch_queue)}) — Book {book_vol}, Ch {chapter}")
                    execute_GLiNER(batch_queue, window_meta, book_vol, chapter)

            if len(batch_queue) > 0:
                print(f"  Final batch (size {len(batch_queue)}) — Book {book_vol}, Ch {chapter}")
                execute_GLiNER(batch_queue, window_meta, book_vol, chapter)

        print(f"  Finished {text_file_path.name}")

    # Deduplication
    deduped_node_data = deduplicate_entities(node_data)

    # LLM descriptions
    if GENERATE_DESCRIPTIONS:
        print(f"\nGenerating entity descriptions via Ollama ({OLLAMA_MODEL})...")
        deduped_node_data = generate_descriptions(deduped_node_data)

        print(f"Generating edge descriptions via Ollama ({OLLAMA_MODEL})...")
        generate_edge_descriptions(edge_data)
    else:
        print("\nSkipping descriptions (GENERATE_DESCRIPTIONS=False)")

    # Output
    if deduped_node_data:
        nodes_df = pd.DataFrame(list(deduped_node_data.values()))
        nodes_df.to_csv("nodes.csv", index=False)
        print(f"Wrote {len(nodes_df)} nodes to nodes.csv")
    else:
        print("No nodes extracted.")

    if edge_data:
        edges_df = pd.DataFrame(list(edge_data.values()))
        edges_df.to_csv("edges.csv", index=False)
        print(f"Wrote {len(edges_df)} edges to edges.csv")
    else:
        print("No edges extracted.")


if __name__ == "__main__":
    LOTR_Extractor()
