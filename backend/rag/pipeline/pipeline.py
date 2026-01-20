from backend.rag.pipeline.classes import DataState, NormalizedMessage, Thread, VectorChunk, TimeAnalytics
from backend.rag.rag_data_utils import find_max_messages, load_raw_messages, normalize_raw_messages, normalize_chunks, create_message_lookup
from backend.chunk_algorithm.chunking import chunk_messages
from backend.rag.pipeline.faiss import create_chunk_embeddings, create_faiss_index
from backend.rag.embedder import get_chunking_embedder, get_rag_embedder

test_phone = "8179134157"
test_name = "Mom"

# Get shared embedder instance (loaded only once)
embedder = get_chunking_embedder()
model_dim = 384

def run_main_pipeline(dataState : DataState) -> DataState: 
  contact_name = dataState.contact_name
  contact_phone = dataState.contact_phone
  
  if dataState.raw_loaded is False: # works 
    num_messages = find_max_messages(contact_phone)
    messages = load_raw_messages(contact_phone, num_messages)
    dataState.raw_messages = messages
    dataState.raw_loaded = True

  if dataState.normalized_ready is False: # works 
    normalized_messages = normalize_raw_messages(dataState.raw_messages)
    dataState.normalized_messages = normalized_messages
    dataState.normalized_ready = True

  if dataState.time_ready is False: 
    pass # ignore for now 
  
  if dataState.threads_ready is False:
    all_chunks = chunk_messages(dataState.raw_messages, embedder) 
    message_lookup = create_message_lookup(all_chunks)
    normalized_chunks = normalize_chunks(all_chunks) 
    dataState.threads = normalized_chunks
    dataState.message_lookup = message_lookup
    dataState.threads_ready = True

  if dataState.rag_ready is False:
    import numpy as np

    chunk_embeddings = create_chunk_embeddings(dataState.threads, dataState.message_lookup, embedder)
    faiss_index = create_faiss_index(dataState.contact_name, dataState.faiss_dim)

    # Convert embeddings to numpy and add to FAISS index
    embedding_matrix = []
    for emb in chunk_embeddings:
      if hasattr(emb, 'cpu'):
        emb_np = emb.cpu().numpy()
      else:
        emb_np = np.array(emb)
      embedding_matrix.append(emb_np.flatten())

    embedding_matrix = np.array(embedding_matrix, dtype='float32')
    faiss_index['index'].add(embedding_matrix)

    dataState.faiss_index = faiss_index
    dataState.rag_ready = True

  return dataState 


def test_pipeline():
    """Test function to run the pipeline and display results."""
    print("\n" + "="*80)
    print("PIPELINE TEST - Data Processing for iMessage RAG System")
    print("="*80)

    # Create DataState
    print(f"\n[1] Initializing DataState...")
    dataState = DataState(
        session_id="test_session_001",
        contact_phone=test_phone,
        contact_name=test_name
    )
    print(f"    Contact: {dataState.contact_name} ({dataState.contact_phone})")

    # Run pipeline
    print(f"\n[2] Running pipeline...")
    result = run_main_pipeline(dataState)

    # Display results
    print("\n" + "="*80)
    print("PIPELINE RESULTS")
    print("="*80)

    print(f"\n[Stage 1] Raw Messages:")
    print(f"  Status: {'✓ LOADED' if result.raw_loaded else '✗ NOT LOADED'}")
    print(f"  Count: {len(result.raw_messages)} messages")
    if result.raw_messages:
        print(f"\n  First message (raw tuple):")
        first_raw = result.raw_messages[0]
        print(f"    {first_raw}")

    print(f"\n[Stage 2] Normalized Messages:")
    print(f"  Status: {'✓ READY' if result.normalized_ready else '✗ NOT READY'}")
    print(f"  Count: {len(result.normalized_messages)} messages")
    if result.normalized_messages:
        print(f"\n  Sample messages (first 3):")
        for i, msg in enumerate(result.normalized_messages[:3]):
            print(f"\n    Message {i+1}:")
            print(f"      ID: {msg.message_id}")
            print(f"      Sender: {msg.sender}")
            print(f"      Time: {msg.timestamp}")
            print(f"      Order: {msg.order_index}")
            print(f"      Content: {msg.content[:60]}...")

    print(f"\n[Stage 3] Time Analytics:")
    print(f"  Status: {'✓ READY' if result.time_ready else '⏸ SKIPPED (not implemented)'}")

    print(f"\n[Stage 4] Threading:")
    print(f"  Status: {'✓ READY' if result.threads_ready else '⏸ IN PROGRESS'}")
    print(f"  Threads: {len(result.threads)}")
    if result.threads:
        print(f"\n  Sample thread (first thread):")
        first_thread = result.threads[0]
        print(f"    Thread ID: {first_thread.thread_id}")
        print(f"    Start: {first_thread.start_time}")
        print(f"    End: {first_thread.end_time}")
        print(f"    Messages in thread: {len(first_thread.message_ids)}")
        print(f"    Duration: {first_thread.metadata.get('duration_seconds', 0):.2f} seconds")

    print(f"\n[Stage 5] Message Lookup:")
    print(f"  Status: {'✓ CREATED' if result.message_lookup else '✗ NOT CREATED'}")
    if result.message_lookup:
        print(f"  Total messages indexed: {len(result.message_lookup)}")

    print(f"\n[Stage 6] RAG System:")
    print(f"  Status: {'✓ READY' if result.rag_ready else '✗ NOT READY'}")
    if result.faiss_index:
        print(f"  FAISS Index: Created")
        print(f"  Dimension: {result.faiss_dim}")

    if result.errors:
        print(f"\n⚠ ERRORS ({len(result.errors)}):")
        for err in result.errors:
            print(f"  - {err}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

    return result


def display_thread_content(thread: Thread, message_lookup: dict, max_messages: int = None):
    """
    Display the full content of a thread with all its messages.

    Args:
        thread: Thread object to display
        message_lookup: Dictionary mapping message_id to message content
        max_messages: Optional limit on number of messages to display
    """
    print("\n" + "─"*80)
    print(f"Thread ID: {thread.thread_id}")
    print(f"Time Range: {thread.start_time} → {thread.end_time}")
    print(f"Duration: {thread.metadata.get('duration_seconds', 0):.2f} seconds")
    print(f"Message Count: {len(thread.message_ids)}")
    print("─"*80)

    messages_to_show = thread.message_ids[:max_messages] if max_messages else thread.message_ids

    for idx, message_id in enumerate(messages_to_show):
        msg_data = message_lookup.get(message_id)
        if msg_data:
            sender = msg_data['sender']
            text = msg_data['text']
            timestamp = msg_data['timestamp']
            print(f"\n  [{idx+1}] {timestamp} | {sender}:")
            print(f"      {text}")

    if max_messages and len(thread.message_ids) > max_messages:
        print(f"\n  ... and {len(thread.message_ids) - max_messages} more messages")

    print("─"*80)


def display_all_threads(dataState: DataState, max_threads: int = 5, max_messages_per_thread: int = 3):
    """
    Display multiple threads with their content.

    Args:
        dataState: DataState object containing threads and message_lookup
        max_threads: Maximum number of threads to display
        max_messages_per_thread: Maximum messages to show per thread
    """
    print("\n" + "="*80)
    print(f"THREAD CONTENT DISPLAY ({len(dataState.threads)} total threads)")
    print("="*80)

    threads_to_show = dataState.threads[:max_threads]

    for idx, thread in enumerate(threads_to_show):
        print(f"\n{'='*80}")
        print(f"THREAD {idx + 1} of {len(threads_to_show)}")
        display_thread_content(thread, dataState.message_lookup, max_messages_per_thread)

    if len(dataState.threads) > max_threads:
        print(f"\n... and {len(dataState.threads) - max_threads} more threads not shown")

    print("\n" + "="*80 + "\n")


def display_message_lookup_stats(message_lookup: dict):
    """
    Display statistics about the message lookup dictionary.

    Args:
        message_lookup: Dictionary mapping message_id to message content
    """
    print("\n" + "="*80)
    print("MESSAGE LOOKUP STATISTICS")
    print("="*80)

    print(f"\nTotal messages indexed: {len(message_lookup)}")

    # Count senders
    sender_counts = {}
    text_lengths = []

    for msg_data in message_lookup.values():
        sender = msg_data['sender']
        sender_counts[sender] = sender_counts.get(sender, 0) + 1
        text_lengths.append(len(msg_data['text']))

    print(f"\nMessages by sender:")
    for sender, count in sender_counts.items():
        print(f"  {sender}: {count} messages ({count/len(message_lookup)*100:.1f}%)")

    print(f"\nMessage length statistics:")
    print(f"  Average: {sum(text_lengths)/len(text_lengths):.1f} characters")
    print(f"  Min: {min(text_lengths)} characters")
    print(f"  Max: {max(text_lengths)} characters")

    # Sample messages
    print(f"\nSample messages (first 3):")
    for idx, (msg_id, msg_data) in enumerate(list(message_lookup.items())[:3]):
        print(f"\n  {idx+1}. ID: {msg_id}")
        print(f"     {msg_data['timestamp']} | {msg_data['sender']}: {msg_data['text'][:60]}...")

    print("\n" + "="*80 + "\n")


def display_thread_statistics(threads: list[Thread]):
    """
    Display aggregate statistics about threads.

    Args:
        threads: List of Thread objects
    """
    print("\n" + "="*80)
    print("THREAD STATISTICS")
    print("="*80)

    print(f"\nTotal threads: {len(threads)}")

    # Calculate statistics
    message_counts = [len(t.message_ids) for t in threads]
    durations = [t.metadata.get('duration_seconds', 0) for t in threads]

    print(f"\nMessages per thread:")
    print(f"  Average: {sum(message_counts)/len(message_counts):.1f}")
    print(f"  Min: {min(message_counts)}")
    print(f"  Max: {max(message_counts)}")

    print(f"\nThread duration:")
    print(f"  Average: {sum(durations)/len(durations):.1f} seconds ({sum(durations)/len(durations)/60:.1f} minutes)")
    print(f"  Min: {min(durations):.1f} seconds")
    print(f"  Max: {max(durations):.1f} seconds ({max(durations)/60:.1f} minutes)")

    # Thread size distribution
    small_threads = sum(1 for c in message_counts if c <= 3)
    medium_threads = sum(1 for c in message_counts if 4 <= c <= 10)
    large_threads = sum(1 for c in message_counts if c > 10)

    print(f"\nThread size distribution:")
    print(f"  Small (1-3 messages): {small_threads} ({small_threads/len(threads)*100:.1f}%)")
    print(f"  Medium (4-10 messages): {medium_threads} ({medium_threads/len(threads)*100:.1f}%)")
    print(f"  Large (>10 messages): {large_threads} ({large_threads/len(threads)*100:.1f}%)")

    print("\n" + "="*80 + "\n")


def test_retrieval_simulation(dataState: DataState, query: str, k: int = 3):
    """
    Simulate a retrieval query (without actual FAISS search, just shows structure).

    Args:
        dataState: DataState object with threads and message_lookup
        query: Query string
        k: Number of results to simulate
    """
    print("\n" + "="*80)
    print("RETRIEVAL SIMULATION")
    print("="*80)

    print(f"\nQuery: \"{query}\"")
    print(f"Top-k: {k}")

    print(f"\nSimulating retrieval from {len(dataState.threads)} threads...")

    # Just show first k threads as example (in real system, FAISS would rank these)
    simulated_results = dataState.threads[:k]

    print(f"\n{'='*80}")
    print(f"SIMULATED RESULTS (showing first {k} threads as example)")
    print("="*80)

    for rank, thread in enumerate(simulated_results, 1):
        print(f"\n[Result {rank}]")
        display_thread_content(thread, dataState.message_lookup, max_messages=5)

    print("\n" + "="*80)
    print("NOTE: This is a simulation. Real retrieval would use FAISS to rank by relevance.")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run the full pipeline test
    result = test_pipeline()

    # Additional detailed displays
    print("\n\n" + "#"*80)
    print("DETAILED ANALYSIS")
    print("#"*80)

    # Show thread statistics
    if result.threads:
        display_thread_statistics(result.threads)

    # Show message lookup stats
    if result.message_lookup:
        display_message_lookup_stats(result.message_lookup)

    # Show actual thread content
    if result.threads and result.message_lookup:
        display_all_threads(result, max_threads=3, max_messages_per_thread=5)

    # Simulate a retrieval
    if result.threads and result.message_lookup:
        test_retrieval_simulation(result, "dinner plans", k=2) 