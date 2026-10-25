

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello world")
print([t.text for t in doc])

