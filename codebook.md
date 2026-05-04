

# Node Attribute
- Node_Type
- Id    (TYPE + LABEL) (EX. EVT_UPRISING_OF_1999)
- Description   (Mostly for events)
- Label
- Document_count
- First_appearance
- Last_appearance
- Source_documents
- Confidence (score)

# Edge Attribute
- Source
- Target
- Label
- Type
- Context
- Source_Document
- Weight

## Process at line 104
1. Check if line matches chapter_pattern, if so, start fresh
2. Fill the window, slides the deque
3. Stride Trigger (Gliner)
4. NER Extract (if applicable)
5. Final NER Extract (for remainder)

node_data -> Dictionary {entity, attributes}