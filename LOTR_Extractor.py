import spacy
import gliner
import regex
import deque
from pathlib import Path

# =============================== CONFIGURATION ===============================

directory_LOTR = Path(r"C:\Users\Azn\Focused\NPS_LOTR_Extractor\LOTR-Extractor\LOTR_Text")

# =============================== PHASE 1 ===============================
node_data = []
edge_data = []
chapter_pattern = regex.compile(r"_Chapter\s\d+_")     # _Chapter 00_
txt_LOTR = ""

# Implement deque to read 50 words and advance 5 words per

# =============================== PHASE 2 ===============================

for text in directory_LOTR.glob("*.txt"):
    print("============ NEXT BOOK ============")
    with open(text, 'r', encoding='cp1252') as file:
        for line in file:
            txt_LOTR += line
    print(txt_LOTR[:500] + "\n...[TRUNCATED]...")

        

# =============================== PHASE 3 ===============================

# =============================== PHASE 4 ===============================