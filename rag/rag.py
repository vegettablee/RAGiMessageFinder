# imports
import os
import sys
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, SimilarityFunction
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# Add data processing directory to path
message_dir = "data/processing"
sys.path.append(message_dir)
import file


def initialize_embedding_model(model_name="multi-qa-MiniLM-L6-cos-v1"):
    """
    Initialize and return the sentence transformer embedding model.

    Args:
        model_name: Name of the pre-trained model to use

    Returns:
        embedder: SentenceTransformer model instance
        dimension: Embedding dimension size
    """
    embedder = SentenceTransformer(model_name, similarity_fn_name=SimilarityFunction.COSINE)
    dimension = 384  # dim-size for multi-qa-MiniLM-L6-cos-v1
    return embedder, dimension


def load_data(subject_phone, subject_name, messages_per_subject=5000, sentences_per_embedding=6):
    """
    Load message data from files and prepare corpus.

    Args:
        subject_phone: Phone number of the subject
        subject_name: Name of the subject
        messages_per_subject: Number of messages to load
        sentences_per_embedding: Number of sentences to group per embedding

    Returns:
        corpus: List of text chunks for embedding
        index_multiplier: Multiplier for index lookups
    """
    file.addToTextFile(subject_phone, messages_per_subject)

    index_multiplier = 1 * sentences_per_embedding
    batch_data = file.getTextFile(sentences_per_embedding)

    corpus = []
    sentences = ""
    for batch in batch_data:
        for sentence in batch:
            sentences += sentence + " "
        corpus.append(sentences)
        sentences = ""

    print("number of text messages : " + str(len(corpus)))
    return corpus, index_multiplier


def create_query_vector(embedder, query_text):
    """
    Convert a query string into an embedding vector.

    Args:
        embedder: SentenceTransformer model instance
        query_text: The query string to embed

    Returns:
        query_embedding: Tensor of shape (1, dimension)
    """
    np_query_embedding = embedder.encode_query(query_text)
    query_embedding = torch.tensor(np_query_embedding, dtype=torch.float32)
    query_embedding = torch.unsqueeze(query_embedding, dim=0)
    return query_embedding


def create_faiss_index(embedder, corpus, dimension):
    """
    Create embeddings for corpus and add them to a FAISS index.

    Args:
        embedder: SentenceTransformer model instance
        corpus: List of text chunks to embed
        dimension: Embedding dimension size

    Returns:
        index: FAISS index containing all embeddings
        embeddings: Tensor of embeddings
    """
    np_embeddings = embedder.encode_corpus(corpus)
    embeddings = torch.tensor(np_embeddings, dtype=torch.float32)

    num_of_vectors = len(np_embeddings)
    print("Numbers of embedding vectors " + str(num_of_vectors))

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("Number of vectors in FAISS : " + str(index.ntotal))
    return index, embeddings




def start():
    # start needs to retrieve the model
    return 0
