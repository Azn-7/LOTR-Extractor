def add_node(type, entity, ID, description, start, end, confidence):
    new_node = {
        "Node_Type"     : type,
        "ID"            : ID,
        "Label"         : entity,
        "Description"   : description,
        "First_Apperance": start,
        "Last_Apperance": end,
        "Confidence"    : confidence
    }
    return new_node

def add_edge(source, target, label, type, context, source_document, weight):
    new_edge = {
        "Source"        : source,
        "Target"        : target,
        "Label"         : label,
        "Type"          : type,
        "Context"       : context,
        "Source_Document": source_document,
        "Weight"        : weight
    }
    return new_edge