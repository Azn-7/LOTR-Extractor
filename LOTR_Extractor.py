import spacy
from gliner import GLiNER
import regex
from collections import deque
from pathlib import Path

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

# Deque (Read 50 words each, advance 5 words at a time)
WINDOW_SIZE = 50
STRIDE = 5

# GLiNER
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
labels = ["node_type", "Id", "description", "appearance_count"]

# Alias Dictionary

# =============================== PHASE 2/3/4 ===============================

for text in directory_LOTR.glob("*.txt"):
    # Initialize
    print("============ NEXT BOOK ============")
    window = deque(maxlen=WINDOW_SIZE)
    word_counter = 0

    # Start reading
    with open(text, 'r', encoding='cp1252') as file:
        for line in file:           # Splits into lines
            # 1. Check if line matches chapter_pattern, if so, start fresh
            if chapter_pattern.search(line):
                window.clear()      # Clean window
                word_counter = 0
                continue

            words = line.split()    # Splits every word and placed in a list
            for word in words:

                # 2. Fill the window
                window.append(word)     # Will automatically remove old words
                word_counter += 1

                # 3. Stride Trigger (Gliner)
                if len(window) == WINDOW_SIZE:  # Is window full?
                    if word_counter % STRIDE == 0:  # Checks if 5 word passed
                        gliner_window = window.join()
                        entities = model.predict_entities(window, labels, threshold=0.5)
                        
                        

    print(f"Finished reading {text.name}")

# =============================== PHASE 3 ===============================

# =============================== PHASE 4 ===============================