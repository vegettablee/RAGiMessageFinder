import chainlit as cl
import faiss
import numpy as np
from chainlit_app.session_state import get_data_state
from backend.rag.embedder import get_rag_embedder
from backend.rag.pipeline.classes import MultiQueryOutput
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
# Get shared embedder instance (loaded only once)
embedder = get_rag_embedder()

top_k = 5  # number of documents to return per result

def route_contact(selected_contacts: list, query: str):
  """
  Route query to selected contacts and retrieve results from their FAISS indexes.

  Args:
      selected_contacts: List of contact dicts with "name" and "phone" keys
      query: Search query string

  Returns:
      doc_results: Dictionary mapping contact_phone -> (threads, message_lookup)
  """
  all_data_states = get_data_state()
  doc_results = {}  # contact_phone : (threads, message_lookup)

  for contact in selected_contacts:
      if contact["name"] in all_data_states.DATA_REGISTRY:
          data_state = all_data_states.DATA_REGISTRY[contact["name"]]
          faiss_index = data_state.faiss_index
          threads = data_state.threads
          message_lookup = data_state.message_lookup

          # Query this contact's index
          results = query_contact(
              query=query,
              k=top_k,
              faiss_index=faiss_index,
              threads=threads,
              message_lookup=message_lookup
          )
          doc_results[contact["phone"]] = results

  return doc_results


def query_contact(query: str, k: int = 5, faiss_index=None, threads=None, message_lookup=None) -> tuple:
  """
  Query a contact's FAISS index.

  Args:
      query: Search query string
      k: Number of results to return
      faiss_index: FAISS index dict with 'index' key
      threads: List of Thread objects
      message_lookup: Dictionary mapping message_id to message content

  Returns:
      Tuple of (retrieved_threads, message_lookup)
  """
  if faiss_index is None or threads is None or message_lookup is None:
      raise ValueError("faiss_index, threads, and message_lookup must be provided")

  # Extract the actual FAISS index from the dict
  index = faiss_index.get('index')

  if index is None:
      print("Warning: FAISS index is None, returning empty results")
      return [], message_lookup

  # Check if index has vectors
  if index.ntotal == 0:
      print("Warning: FAISS index is empty, returning empty results")
      return [], message_lookup

  try:
      # Create query embedding
      query_embedding = embedder.encode(query, convert_to_tensor=True)

      # Convert to numpy array for FAISS (ensure float32 and contiguous)
      if hasattr(query_embedding, 'cpu'):
          query_vector = query_embedding.cpu().numpy()
      else:
          query_vector = np.array(query_embedding)

      query_vector = query_vector.astype('float32').reshape(1, -1)

      # Make sure it's C-contiguous for FAISS
      if not query_vector.flags['C_CONTIGUOUS']:
          query_vector = np.ascontiguousarray(query_vector)

      # Limit k to available vectors
      k_actual = min(k, index.ntotal)

      # Search
      distances, indices = index.search(query_vector, k_actual)

      # Retrieve results (validate indices)
      results = []
      for idx in indices[0]:
          if 0 <= idx < len(threads):
              results.append(threads[idx])
          else:
              print(f"Warning: Invalid index {idx}, skipping")

      return results, message_lookup

  except Exception as e:
      print(f"Error during FAISS search: {e}")
      import traceback
      traceback.print_exc()
      return [], message_lookup

def multi_query_translation(query : str) -> list[str]: 
    # be able to sort by time 
    query_categories = ["Temporal", "Information-Retrieval", "Creative"]
    print("User query : ", query)
    # for sentiment analysis along with 
    # goals for prompt, classify in 4 categories with boundaries for each one 
    # for temporal queries, try different time translations that mean the same thing 
    # for information retrieval, like about an event, try different translations but be as direct as possible 
    # for queries not pertaining to either category, come up with different translations that are creative 
    # 5 different queries will be created
    # there will be one field called "category" where it will contain one of the three categories
    # and then there will be another field called "queries" where it is linked to an array of strings 
    
    # i want the json in the form of an array with every 
    template = f"""
You are a query translator for a RAG system built on iMessage conversational data.

Your task: Generate 5 semantically equivalent variations of the user's query below.

CLASSIFICATION RULES:
- "Temporal": Query asks WHEN something happened (e.g., "When did...", "Last time...", "Recent...")
- "Information-Retrieval": Query seeks specific facts, people, or things (e.g., "Who...", "What is...", "Find...")
- "Creative": Query is open-ended, exploratory, or opinion-based (e.g., "Why...", "How can I...", "What are good...")

GENERATION RULES:
1. Keep the SAME entities (people, places, things mentioned)
2. Keep the SAME intent and time frame
3. Only vary the phrasing/wording and/or pronouns if they are semantically similar.
4. Goal: Maximize retrieval coverage by using different phrasings that might match different ways the information appears in messages

USER QUERY: "{query}"

Return ONLY valid JSON in this exact format:
{{{{
  "category": "<Temporal | Information-Retrieval | Creative>",
  "queries": [
    "<variation 1>",
    "<variation 2>",
    "<variation 3>",
    "<variation 4>",
    "<variation 5>"
  ]
}}}}
"""

    parser = PydanticOutputParser(pydantic_object=MultiQueryOutput)
    format_instructions = parser.get_format_instructions()
    llm = ChatOpenAI(temperature=0)

    response = llm.invoke(template)

    print(response.content)

    parsed = parser.parse(response.content)

    category = parsed.category   # "Temporal"
    queries = parsed.queries


    return category, queries 
