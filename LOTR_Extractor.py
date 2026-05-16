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
        return nd, {}

# ==============================================================================
# =============================== CONFIGURATION ================================
# ==============================================================================

directory_LOTR = Path(r"C:\Coding\LOTR-Extractor\LOTR_Text")
TEST_MODE = False        # Set to False to process all LOTR_Text files
GENERATE_DESCRIPTIONS = False  # Set to False to skip Ollama and write [N/A] descriptions

OLLAMA_MODEL  = "llama3.1:8b"
OLLAMA_URL    = "http://localhost:11434/api/chat"
MIN_REF_COUNT       = 2   # Minimum occurrences for PERSON, LOCATION, ORGANIZATION
MIN_EVENT_REF_COUNT = 15  # Higher bar for EVENT — filters vague one-off extractions

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
labels = [
    "named person or character",
    "named group, council, or people",
    "named place, city, region, or landmark",
    "named historical event or battle",
]

# Maps verbose GLiNER label text back to the short Node_type stored in node_data
LABEL_NORMALIZE = {
    "named person or character":              "PERSON",
    "named group, council, or people":        "ORGANIZATION",
    "named place, city, region, or landmark": "LOCATION",
    "named historical event or battle":       "EVENT",
}
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
    # Aliases found in full-text audit
    "the poor old gaffer": "Old Gaffer Gamgee",
    "dad": "Old Gaffer Gamgee",
    "the old took": "Old Took",
    "the old wizard": "Gandalf the Grey",
    "the ring-bearer": "Frodo Baggins",
    "stick-at-naught strider": "Aragorn Son Of Arathorn",
    "nine servants of the lord of the rings": "Ringwraiths",
    # Sauron epithets — map verbose canonical and all epithets to clean key
    "sauron the base master of treachery": "Sauron",
    "the dark lord": "Sauron",
    "the nameless enemy": "Sauron",
    "the necromancer": "Sauron",
    # Accent-stripped / misextracted characters
    "owyn": "Éowyn",
    # Wordy extractions → clean canonical
    "forlong": "Forlong The Fat",
    "meriadoc of the shire": "Merry Brandybuck",
    "lords of the house of eorl the young": "The Rohirrim",
}

type_overrides = {
    # Locations misclassified as PERSON by GLiNER
    "lothlórien":  "LOCATION",
    "khazad-dûm":  "LOCATION",
    "amon sûl":    "LOCATION",
    "o lórien":    "LOCATION",
    "caradhras":   "LOCATION",
    "sméagol":     "PERSON",
    "old butterbur": "PERSON",
    # Misclassified types found in 3-book audit
    "owyn":                              "PERSON",    # Éowyn, accent-stripped
    "eorl":                              "PERSON",    # Eorl the Young
    "forlong":                           "PERSON",    # Forlong the Fat
    "paladin of the shire of the halflings": "PERSON",  # Paladin Took II
    "meriadoc of the shire":             "PERSON",    # Merry Brandybuck
    "rammas":                            "LOCATION",  # Rammas Echor
    "halifirien":                        "LOCATION",  # Beacon-hill, Gondor/Rohan border
    "the fords":                         "LOCATION",  # Fords of Isen
    "muil":                              "LOCATION",  # Emyn Muil
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
    # Titles used as standalone entities
    "sir", "lord", "lady", "king", "queen", "master", "mistress",
    # Generic PERSON phrases (possessives, roles, fragments)
    "his companions", "his three companions", "their spirits", "young lord",
    "my darling", "her lover", "dark heads", "alone", "wraith", "lad",
    "carpenter", "matriarch", "immortal maiden", "keeper", "firebrand",
    "bright", "starling", "three large trolls",
    # Generic "The X" phrases
    "the forest", "the hedge", "the mirror", "the watch", "the flood",
    "the porch", "the trail", "the place", "the summit", "the surface",
    "the brim", "the drift", "the glen", "the spot", "the pile", "a pile",
    "the ceiling", "the chasm", "the gloom", "the blaze", "the chase",
    "the alarm", "the ferry", "the watching eyes",
    # Generic "Old X" / title phrases
    "old troll", "old wives", "old winyards", "king under the mountain",
    "queen of the stars",
    # Fragments (short garbage)
    "ken", "loo", "s.", "g3", "am", "ii", "hen", "hee", "dol", "tim",
    # Artifacts / incomplete
    "mr.", "announcement_", "morning_",
    # Misclassified
    "bucklanders", "shire-folk", "chief table", "boats",
    # Generic "The X" / "An X" persons (descriptors, not proper names)
    "the cow", "the dog", "the figure", "the innkeeper", "the singer",
    "the thief", "the sun", "the messenger", "the best hobbit",
    "an elven-maid", "an orc", "the firstborn",
    # Generic "The X" locations
    "the house", "the table", "the passage", "the wood",
    # Generic organization phrases
    "the others", "my young friends", "my people", "some of my kindred",
    "her maidens", "his followers", "wise ones", "big and little",
    "elf-friend", "black rulers", "the audience", "the remnant",
    "rowans", "shadow", "the local inhabitants", "countrymen", "gardeners",
    "elders", "companions", "hunters", "exiles", "the scouts", "northern kindred",
    # High-ref generic descriptors
    "a collection of local hobbits", "a strong company of orcs",
    "a squint-eyed ill-favoured fellow",
    # Short fragments
    "home", "séa", "edro", "ford", "b.b.",
    # Vague events
    "party", "flood", "dark times",
    # Artifacts misclassified as persons
    "the silmaril",
    # Short fragments (3-book audit)
    "day", "howe", "boys", "ape", "boro", "both", "fey", "hoom", "naur",
    "sun", "swan", "thou", "yéni", "yule", "garn", "nick",
    "grã", "cã", "dãºnedain of the north",   # encoding artifacts
    # Vague events
    "moot", "shire-reckoning", "new year", "sortie",
    # Generic PERSON descriptors
    "the servant of the prince", "the head of the orc-company",
    "the landlord", "the boss", "the tracker", "the woman", "the host",
    "the ostler", "the little wretch", "the scout", "the little dog",
    "the slave-driver", "boromir_you", "the unnamed", "the guide",
    "the healer", "the haggard king", "the tree-killer", "the evil voice",
    # Generic ORGANIZATION phrases
    "my scouts and watchers", "dear friends of the shire", "king of rohan",
    "ill-mannered children", "others", "his people", "messengers",
    "ye people of the tower of guard", "orcses", "cruel peoples", "shirefolk",
    "the watchmen", "the living", "elder people", "your neighbours",
    "his own people", "healers", "troop-leaders", "the pursuers",
    "orc-voices", "forayers", "hobbit-lordlings", "close-serried companies",
    "your servants", "hero of the age", "green-clad warriors", "many peoples",
    "his counsel", "your people", "the builders of old", "his eye",
    "the rammers", "trees", "the lembas", "the captains", "thain of the shire",
    # Generic LOCATION phrases
    "the common room", "the cleft", "the gateway", "the head of the rapids",
    "the plain", "the way", "the mouth of the gully", "the nuncheon",
    "the opening", "the path", "the star-glass",
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
            raw_text = entity_dict["text"].replace(" _", " ").replace("_ ", " ").strip("_").strip()

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
                if name in type_overrides:
                    node_data[name]['Node_type'] = type_overrides[name]
                window_canonical_names.append(name)
                continue

            # Create new entity
            attribute = {}
            raw_label = LABEL_NORMALIZE.get(entity_dict['label'], entity_dict['label'])
            node_type = type_overrides.get(name, raw_label)
            attribute['Node_type'] = node_type
            if node_type == "EVENT":
                attribute['ID'] = ("EVT" + " " + name).replace(" ", "_").upper()
            else:
                attribute['ID'] = (node_type[:3] + " " + name).replace(" ", "_").upper()
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
    invalid_keys = set()

    system_prompt = (
        "You are a literary analyst for Lord of the Rings.\n"
        "For each entity, decide: is this a real, specifically named entity from Lord of the Rings "
        "(a named character, place, group, or historical event)? "
        "Or is it too vague, generic, or not a proper named noun?\n\n"
        "If it is real and specific, write a single concise sentence describing it.\n"
        "Use this exact format — one line per entity:\n"
        "entity_key: VALID One-line description.\n"
        "entity_key: INVALID\n\n"
        "Examples:\n"
        "bilbo baggins: VALID A well-to-do hobbit of the Shire who lives at Bag End and is known for his unexpected adventure.\n"
        "the old man: INVALID\n"
        "the war of the ring: VALID The great conflict between the Free Peoples of Middle-earth and Sauron's forces of Mordor.\n"
        "the meeting: INVALID"
    )

    for i in tqdm(range(0, len(entities), DESC_BATCH), desc="  Validating and describing entities", unit="batch"):
        batch = entities[i : i + DESC_BATCH]
        lines = []
        for key, attrs in batch:
            lines.append(
                f"Name: {key}\n"
                f"Type: {attrs.get('Node_type', 'ENTITY')}\n"
                f"Context: {attrs['_context']}"
            )
        user_msg = "Validate and describe these entities:\n\n" + "\n\n---\n\n".join(lines)

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
            key_part, rest = line.split(':', 1)
            canonical = key_part.strip().lower()
            rest = rest.strip()
            if not rest or canonical not in node_data:
                continue
            if rest.upper().startswith('INVALID'):
                invalid_keys.add(canonical)
            elif rest.upper().startswith('VALID'):
                node_data[canonical]['Description'] = rest[5:].strip()

    for key in invalid_keys:
        node_data.pop(key, None)

    if invalid_keys:
        print(f"  Removed {len(invalid_keys)} entities flagged as invalid: {', '.join(sorted(invalid_keys))}")

    for attrs in node_data.values():
        attrs.pop('_context', None)

    return node_data


def generate_edge_descriptions(edge_data: dict, deduped_nodes: dict) -> dict:
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
            src_label = deduped_nodes.get(attrs["Source"], {}).get("Label", attrs["Source"])
            tgt_label = deduped_nodes.get(attrs["Target"], {}).get("Label", attrs["Target"])
            src_type = deduped_nodes.get(attrs["Source"], {}).get("Node_type", "ENTITY")
            tgt_type = deduped_nodes.get(attrs["Target"], {}).get("Node_type", "ENTITY")
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


def _remap_edges(edge_data: dict, key_map: dict) -> None:
    """
    Update edge Source/Target keys to their post-deduplication canonical keys.
    Edges whose endpoints collapse to the same canonical key (self-loops) are
    removed. Edges that map to the same canonical pair are merged by summing
    Weight and keeping the highest-confidence context.
    """
    for old_key in list(edge_data.keys()):
        edge = edge_data[old_key]
        new_src = key_map.get(edge["Source"], edge["Source"])
        new_tgt = key_map.get(edge["Target"], edge["Target"])

        if new_src == new_tgt:
            del edge_data[old_key]
            continue

        edge["Source"] = new_src
        edge["Target"] = new_tgt
        new_key = frozenset({new_src, new_tgt})

        if new_key == old_key:
            continue

        if new_key in edge_data:
            edge_data[new_key]["Weight"] += edge["Weight"]
        else:
            edge_data[new_key] = edge

        del edge_data[old_key]


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
    deduped_node_data, key_map = deduplicate_entities(node_data)
    _remap_edges(edge_data, key_map)

    # Drop nodes below minimum reference count (stricter threshold for EVENTs)
    deduped_node_data = {
        k: v for k, v in deduped_node_data.items()
        if v.get("Reference_Count", 1) >= (
            MIN_EVENT_REF_COUNT if v.get("Node_type") == "EVENT" else MIN_REF_COUNT
        )
    }

    # Purge edges whose endpoints were removed
    valid_keys = set(deduped_node_data.keys())
    for key in list(edge_data.keys()):
        edge = edge_data[key]
        if edge["Source"] not in valid_keys or edge["Target"] not in valid_keys:
            del edge_data[key]

    # Refresh edge labels to reflect corrected node types after dedup/type_overrides
    for edge in edge_data.values():
        src_type = deduped_node_data.get(edge["Source"], {}).get("Node_type", "UNKNOWN")
        tgt_type = deduped_node_data.get(edge["Target"], {}).get("Node_type", "UNKNOWN")
        edge["Label"] = f"{src_type}-{tgt_type}"

    # LLM descriptions
    if GENERATE_DESCRIPTIONS:
        print(f"\nValidating and describing entities via Ollama ({OLLAMA_MODEL})...")
        deduped_node_data = generate_descriptions(deduped_node_data)

        # Re-clean edges — Ollama may have removed nodes flagged as invalid
        valid_keys = set(deduped_node_data.keys())
        for key in list(edge_data.keys()):
            edge = edge_data[key]
            if edge["Source"] not in valid_keys or edge["Target"] not in valid_keys:
                del edge_data[key]

        print(f"Generating edge descriptions via Ollama ({OLLAMA_MODEL})...")
        generate_edge_descriptions(edge_data, deduped_node_data)
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
