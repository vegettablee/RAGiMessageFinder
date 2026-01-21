# imports
import os
import sys
import numpy as np
import torch
import faiss
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from backend.rag.embedder import get_rag_embedder

import sys
import os 

subject_phone = "9365539666"
subject_name = "Paris"


async def initialize_rag_pipeline():
    # load qachain for queries, don't load any databases, this only happens when a contact is added and verified 
    embedder = get_rag_embedder()
    dimension = 384  # dim-size for multi-qa-MiniLM-L6-cos-v1

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.5,
        streaming=True  # Important for Chainlit streaming
    )

    # Return initialized components (no data loaded yet)
    qa_chain = {
        'embedder': embedder,
        'dimension': dimension,
        'llm': llm
    }

    return qa_chain


def initialize_embedding_model():
    """Legacy function - use get_rag_embedder() instead."""
    embedder = get_rag_embedder()
    dimension = 384
    return embedder, dimension


def create_query_vector(embedder, query_text):
    np_query_embedding = embedder.encode_query(query_text)
    query_embedding = torch.tensor(np_query_embedding, dtype=torch.float32)
    query_embedding = torch.unsqueeze(query_embedding, dim=0)
    return query_embedding

def start():
    # start needs to retrieve the model
    return 0
