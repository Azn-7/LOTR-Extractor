import spacy
from gliner import GLiNER
import regex
from collections import deque
from pathlib import Path
import itertools
import torch
import Extractor_Utils

# ==============================================================================
# =============================== CONFIGURATION ===============================
# ==============================================================================

directory_LOTR = Path(r"C:\Users\Azn\Focused\NPS_LOTR_Extractor\LOTR-Extractor\LOTR_Text")

# ==============================================================================
# =============================== INITALIZATION ===============================
# ==============================================================================

# Data Variables
node_data = {}  # {ID:Attribute} {str:dir}
edge_data = {}
valid_characters = []

# Patterns
chapter_pattern = regex.compile(r"_Chapter\s\d+_")     # _Chapter 00_

# Apperance Tracker
book_vol = 0
chapter = 0

# OTHER
txt_LOTR = ""
gliner_window = ""

# Deque (Read 50 words each, advance 5 words at a time)
WINDOW_SIZE = 50
STRIDE = 5

# GLiNER
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1").to('cuda')
labels = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT"]
batch_queue = []
all_batch_results = []  # List of lists of dictionaries
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
}

exceptions = [
    # Pronouns
    "he", "him", "they", "she", "her", "i", "you", "me", "my", "his", "your", "it", "were",
    "he escorted her",
    # Generic roles
    "local legend", "miller", "the miller", "friend", "stranger", "the stranger",
    "visitor", "host", "their host", "person", "somebody", "young fellow",
    "old man", "an old man", "the old man", "people", "some people", "other people",
    "all and sundry", "his guests", "everyone", "cooks", "postman",
    "voluntary assistant postmen", "knowledgeable", "old", "gross",
    # Artifacts / incomplete
    "mr.", "announcement_", "morning_",
    # Misclassified
    "bucklanders", "shire-folk", "chief table", "boats",
]

# ==============================================================================
# ==============================================================================
# ==============================================================================

def execute_GLiNER(batch_queue, book_vol, chapter):
    global edge_data, valid_characters

    # 1. Grab raw data
    batch_results = model.inference(batch_queue, labels, threshold=0.5, batch_size=len(batch_queue))
    # Output: List of list of Dictionaries

    #  ================================ TODO: Work on Nodes first ================================

    for i, window_entities in enumerate(batch_results):
        for entity_dict in window_entities:
            name = entity_dict["text"].lower()

            # Catch false entities
            if name in exceptions:
                continue

            # Update existing entity
            if name in valid_characters:
                new_attribute = node_data[name].values
                new_attribute['Last_Apperance'] = entity_dict['end']
                new_attribute['Reference_Count'] += 1
                node_data[name] = new_attribute
                continue

            if name in alias:
                new_attribute = node_data[name].values
                new_attribute['Last_Apperance'] = entity_dict['end']
                new_attribute['Reference_Count'] += 1
                node_data[name] = new_attribute
                continue

            # Create new entity
            attribute = {}
            attribute['Node_type'] = entity_dict['label']
            attribute['ID'] = (entity_dict['label'][:3] + " " + entity_dict["text"]).replace(" ", "_").upper()
            attribute['Description'] = '[N/A]'
            if name in alias:
                attribute['Label'] = alias[name]
            else:
                attribute['Label'] = entity_dict["text"]
            attribute['Reference_Count'] = 1
            attribute['First_Apperance'] = f"Book {book_vol}, Chapter {chapter}"
            attribute['Last_Apperance'] = f"Book {book_vol}, Chapter {chapter}"
            attribute['Confidence'] = round(entity_dict['score'], 2)

            valid_characters.append(name)
            node_data[name] = attribute

    #  ================================ TODO: Work on Edges last ================================
            # unique_chars = sorted(set(valid_characters))
        # if len(unique_chars) >= 2:
        #     edges = itertools.combinations(unique_chars, 2) # pair[0] = person      pair[1] = person
        #     for pair in edges:
        #         source = pair[0]
        #         target = pair[1]
        #         label = '[N/A]'
        #         context = batch_queue[i]
        #         weight = 1

        #         add_edge(source, target, label, type, context, source_document, weight)

    # 4. Cleanup
    batch_queue.clear()
    return

def LOTR_Extractor():
    # for text in directory_LOTR.glob("*.txt"): # Keep for recursive NER per file
    # =================== Initialize ===================
    text = r"example.txt"
    print("\n============ READING BOOK ============")
    window = deque(maxlen=WINDOW_SIZE)
    word_counter = 0
    # ==================================================

    with open(text, 'r', encoding='cp1252') as file:
        # Start reading
        for line in file:
            if chapter_pattern.search(line):
                window.clear()
                word_counter = 0
                chapter += 1
                continue
            words = line.split()    # Splits every word and placed in a list

            for word in words:
                window.append(word)     # Will automatically remove old words
                word_counter += 1
                if len(window) == WINDOW_SIZE:
                    if word_counter % STRIDE == 0:  # Checks if 5 word passed
                        gliner_window = " ".join(window)
                        batch_queue.append(gliner_window)    

            if len(batch_queue) >= BATCH_SIZE:  # Prevents stack overflow
                print(f"Executing batch! Batch size is {len(batch_queue)}")
                execute_GLiNER(batch_queue, book_vol, chapter)

        if len(batch_queue) > 0:
            print(f"Executing last batch! Batch size is {len(batch_queue)}")
            execute_GLiNER(batch_queue, book_vol, chapter)
    print(f"Finished reading {text}")

    # =================== OUTPUT ===================
    # with open("edges.txt", "w") as file: 
    #     if edge_data:  
    #         for edge in edge_data:
    #             file.write(str(edge) + "\n")
    with open("nodes.txt", "w") as file:
        if node_data:
            file.write(",".join(next(iter(node_data.values())).keys()) + "\n")     # Header
            for node in node_data.values():
                file.write(",".join(str(value) for value in node.values()) + "\n")  # Nodes
                
    # =============================================

if __name__ == "__main__":
    LOTR_Extractor()

# =============================== PHASE 3 ===============================

# =============================== PHASE 4 ===============================