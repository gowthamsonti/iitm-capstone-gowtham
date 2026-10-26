
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Downloaded ccached at :', reranker.model.config._name_or_path)
