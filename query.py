import spacy

nlp = spacy.load("en_core_web_sm")

def process_query(query):
    doc = nlp(query)
    objects = [token.text for token in doc if token.pos_ == "NOUN"]
    return objects
