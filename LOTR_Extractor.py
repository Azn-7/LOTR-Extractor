import spacy
from gliner import GLiNER
import regex
from collections import deque
from pathlib import Path
import itertools

# =============================== CONFIGURATION ===============================

directory_LOTR = Path(r"C:\Users\Azn\Focused\NPS_LOTR_Extractor\LOTR-Extractor\LOTR_Text")

# =============================== PHASE 1 ===============================

# Data Variables
node_data = []
edge_data = []

# Patterns
chapter_pattern = regex.compile(r"_Chapter\s\d+_")     # _Chapter 00_

# OTHER
txt_LOTR = ""
gliner_window = ""

# Deque (Read 50 words each, advance 5 words at a time)
WINDOW_SIZE = 50
STRIDE = 5

# GLiNER
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1").to("cuda")
labels = ["person", "description"]
batch_queue = []
all_batch_results = []  # List of lists of dictionaries
BATCH_SIZE = 64

# Alias Dictionary
alias = {}
# WILL BE INCLUDED LATER

# =============================== PHASE 2/3/4 ===============================

def add_node(entity):
    global node_data
    new_node = {
        "node_type": "",
        "Label": entity,
        "ID": "",
        "Description": "",
    }
    node_data.append(new_node)
    return

def add_edge(pair):
    global edge_data
    new_edge = {
        "Source": pair[0],
        "Target": pair[1],
        "Label": "",
        "Type": "Undirected",
        "Context": "",
        "Weight": 1
    }
    edge_data.append(new_edge)
    return

def execute_GLiNER():
    global batch_queue, edge_data

    # 1. Grab raw data
    batch_results = model.batch_predict_entities(batch_queue, labels, threshold=0.5)
    # Output: List -> List -> Dictionaries
    # Example: [] -> [ {"text": "Frodo", "label": "person", "score": 0.99} , ...]

    # 2. Loop through the list of list
    for window_entities in batch_results:
        valid_characters = []

    # 3. Loop through the list of dictionaries
        for entity_dict in window_entities:
            if entity_dict["label"] == "person":
                raw_name = entity_dict["text"]

                raw_name = raw_name.lower()
                # Alias Check
                if raw_name in alias:
                    raw_name = alias[raw_name]
                valid_characters.append(raw_name)
                add_node(raw_name)
        unique_chars = sorted(set(valid_characters))
        if len(unique_chars) >= 2:
            edges = itertools.combinations(unique_chars, 2)
            for pair in edges:
                add_edge(pair)

    # 4. Cleanup
    batch_queue.clear()
    return

def LOTR_Extractor():
    for text in directory_LOTR.glob("*.txt"):
        # Initialize
        print("============ NEXT BOOK ============")
        window = deque(maxlen=WINDOW_SIZE)
        word_counter = 0

        # Start reading
        with open(text, 'r', encoding='cp1252') as file:
            for line in file:
            # 1. Check if line matches chapter_pattern, if so, start fresh
                if chapter_pattern.search(line):
                    window.clear()
                    word_counter = 0
                    continue

                words = line.split()    # Splits every word and placed in a list
                for word in words:

            # 2. Fill the window, slides the deque
                    window.append(word)     # Will automatically remove old words
                    word_counter += 1

            # 3. Stride Trigger (Gliner)
                    if len(window) == WINDOW_SIZE:
                        if word_counter % STRIDE == 0:  # Checks if 5 word passed
                            gliner_window = " ".join(window)
                            batch_queue.append(gliner_window)
            # 4. NER Extract (if applicable)         
                if len(batch_queue) >= BATCH_SIZE:  # Prevents stack overflow
                    execute_GLiNER()
            # 5. Final NER Extract (for remainder)
            if len(batch_queue) > 0:
                execute_GLiNER()
        
    print(f"Finished reading {text.name}")

if __name__ == "__main__":
    LOTR_Extractor()

# =============================== PHASE 3 ===============================

# =============================== PHASE 4 ===============================