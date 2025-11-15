# this file orchestrates the chunking algorithm by connecting all refactored modules

from sentence_transformers import SentenceTransformer, SimilarityFunction
from data.processing.test_data import message_chunks
from chunk_algorithm.topic_shift import compute_topic_shifts
from chunk_algorithm.micro_thread import compute_microthreads
from chunk_algorithm.remainder import append_remainders

MAX_SECION_TIME = 30 # max amount of time two texts must stay within
LINE_BURST_TIME = 5 # max amount of time for line bursts

def chunk_messages(messages=list, model=None): # pass in embedding model for similarity
  # messages is a list in the format :
  # ('2024-10-08 01:31:13', '', '', 'OOUU it's so fun seeing how other people edit stuff', '+19365539666')

  # compute topic shifts and get valid chunks, this needs to be called with a set amount of messages to get topic shifts
  # TO DO : add logic that separates the messages into chunks of like 100-1000, then compute topic_shifts 
  # once topic shifts are computed, then find the micro-threads, otherwise if ran with too few messages, the algorithm will break
  validated_chunks, text_total, similar_texts, burst_total, b_similar_counter = compute_topic_shifts(messages, model)

  # this is temporary, focus on the part of the 
  # return 
  # Step 2: Flatten validated_chunks to extract individual bursts
  # validated_chunks structure: list of sections, each section has bursts or tuples of merged bursts
  all_bursts = []
  for section in validated_chunks:
    for item in section:
      if isinstance(item, tuple):  # merged bursts: (burst1, burst2)
        # Flatten the tuple - merge both bursts into a single burst
        merged_burst = item[0] + item[1]
        all_bursts.append(merged_burst)
      else:  # individual burst
        all_bursts.append(item)

  # Step 3: Process each adjacent burst pair completely (microthreads + remainders)
  # Note: Process ONE pair at a time, not accumulate all then process
  all_final_threads = []

  for i in range(len(all_bursts) - 1):  # iterate through adjacent burst pairs
    valid_threads, remainders = compute_microthreads(all_bursts[i], all_bursts[i + 1], model)

    # Immediately append remainders for this burst pair
    final_threads = append_remainders(valid_threads, remainders, model)
    all_final_threads.extend(final_threads)

  return all_final_threads


# Main execution: initialize model and run tests
if __name__ == "__main__":
  embedder = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.COSINE)
  counter = 0 
  for message_chunk in message_chunks:
    if counter == 2: 
      break
    print("\n" + "="*80)
    print("Processing new message chunk...")
    print("="*80)
    final_threads = chunk_messages(message_chunk, embedder)
    print("\nProcessing complete!")