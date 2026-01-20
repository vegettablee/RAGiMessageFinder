from backend.rag.pipeline.classes import Thread
import faiss

def create_faiss_index(contact_name: str, dim: int = 384) -> dict:
    # Create Flat L2 index (no IDMap needed - we use sequential indices)
    index = faiss.IndexFlatL2(dim)

    return {
        "contact_name": contact_name,
        "index": index
    }

def create_chunk_embeddings(threads: list[Thread], message_lookup: dict[str, dict], embedder):
    """
    Create embeddings for each thread.

    Args:
        threads: List of Thread objects with message_ids
        message_lookup: Dictionary mapping message_id -> {sender, text, timestamp}
        embedder: SentenceTransformer model for creating embeddings

    Returns:
        List of embeddings (numpy arrays or tensors), one per thread
    """
    embeddings = []

    for thread_idx, thread in enumerate(threads):
        # Retrieve message content using message_ids from the thread
        thread_text = []
        for message_id in thread.message_ids:
            msg_data = message_lookup.get(message_id)
            if msg_data:
                sender = msg_data['sender']
                text = msg_data['text']
                # Format as "sender: text" for better context
                formatted_msg = f"{sender}: {text}"
                thread_text.append(formatted_msg)

        # Join all messages in thread with newlines to create a single document
        thread_document = "\n".join(thread_text)

        # Create embedding for the entire thread as one document
        thread_embedding = embedder.encode(thread_document, convert_to_tensor=True)
        embeddings.append(thread_embedding)

    if len(embeddings) != len(threads): # check if they are the same length, if not, then we have an indexing issue
        return None
    else:
        return embeddings
  