## Imports 
To have a functional RAG pipeline, these need to be imported, library itself is subject to change
1. Embedding model
Chosen : __OpenAI Embeddings__
-- This embedding model encodes text into embeddings that have semantic meaning, which is used directly to encode the 
-- data source/text so that it can be queried using these vectors via cosine similarity
2. LLM Model  
Chosen : __OpenAI__ 
-- OpenAI, Claude, other popular LLMs
-- Transformer models from HuggingFace and Bert are solid choices as well, depending on the task 
3. Vector Similarity/Search 
Chosen : __numPy__ 
-- Any library can be used that supports vector operations like numPy 
-- faiss is good for larger datasets
4. Data Handling 
Chosen : __pandas__ 
-- Any library that has data functions for cleaning/manipulating datasets like pandas or just basic json

##