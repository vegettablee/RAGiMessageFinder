from sentence_transformers import SentenceTransformer, SimilarityFunction
from data.processing.test_data import message_chunks
from chunk_algorithm.topic_shift import compute_topic_shifts
from chunk_algorithm.micro_thread import compute_microthreads
from chunk_algorithm.remainder import append_remainders
from chunk_algorithm.similarity import compute_text_similarity, THREAD_TEXT_SIMILARITY_THRESHOLD
import torch
import math
from datetime import datetime
from data.processing.test_data import message_chunks

embedder = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.COSINE)

# this file holds the algorithm that brute forces the micro-thread extraction
# using this to generate 

def bruteforce_microthreads(burst1, burst2, model): 

  microthreads, global_sum = compute_max(burst1, burst2, model) # when comparing all burst1 messages against burst 2 and finding individual threads, keep a sum

  return microthreads, global_sum 


def compute_max(burst1, burst2, model): 
  max_threads = [] # keep track of the micro-threads that go above the maximum 
  max_sum = 0 

  # keep track of a set of seen messages and their score? 
  # if we calculate a score that uses the same 
  all_messages = burst1 + burst2
  times = list([datetime.strptime(message[0],  '%Y-%m-%d %H:%M:%S') for message in all_messages])
  times_tensor = torch.tensor([t.timestamp() for t in times])
  sorted_indices = torch.argsort(times_tensor)
  time_sorted = []  

  for i in sorted_indices: 
    time_sorted.append(all_messages[i]) 
  
  messages = time_sorted

  start_idx = 0
  end_idx = len(messages)

  order = find_optimal_order(messages, start_idx, end_idx, model)

  return max_threads, max_sum

def find_optimal_order(all_messages, start_idx, end_idx, model): 
  max_permutations = math.pow(2, end_idx - start_idx) - 1
  optimal_order = []
  highest_score = 0 
  used = []
  for i in range(start_idx, end_idx):
    if i == end_idx - 1: # don't compute subsequences for indices that have less than a 2 
      continue
    print(f"Computing subsequences for index {start_idx} - {end_idx}")
    sequence = all_messages[start_idx:end_idx]
    threads = subsequences(sequence, model=model)

    scores = torch.tensor([pair[1] for pair in threads])
    highest_score_idx = torch.argsort(scores)[-1].item()

    best_current_thread = threads[highest_score_idx][0]
    best_current_score = threads[highest_score_idx][1]
    used.append({
      "micro_thread" : best_current_thread,
      "score" : best_current_score
    })

    print(f"\n{'='*60}")
    print(f"Range [{start_idx}:{end_idx}] | Score: {best_current_score:.4f} | Thread length: {len(best_current_thread)}")
    print(f"{'='*60}")
    for msg in best_current_thread:
      sender = msg[1] if msg[1] else "Other"
      print(f"  [{msg[0]}] {sender}: {msg[3]}")

    start_idx += 1
  counter = 0
  global_score = 0

    

# helper function for find_optimal_order, computes all non-contiguous subsequences with length >= 2
# all subsequences start with the first element
def subsequences(seq, start=0, current=None, model=None):
  if current is None:
    # Initialize with first element, then process rest of sequence
    if len(seq) < 2:
      return []
    return subsequences(seq, start=1, current=[seq[0]], model=model)

  if start == len(seq):
    if len(current) >= 2:
      return [(current, weighted_thread_score(current, model))]
    else:
      return []

  include = subsequences(seq, start + 1, current + [seq[start]], model)
  exclude = subsequences(seq, start + 1, current, model)
  return include + exclude

    
def weighted_thread_score(thread, model):
  total_score = sum_microthread(thread, model)
  weight_score = total_score / len(thread) # divide by the length to normalize the length
  return weight_score

def sum_microthread(thread, model): 
  sum = 0
  for idx, message in enumerate(thread): 
    if idx == len(thread) - 1: 
      break
    first = message[3]
    second = thread[idx + 1][3]
    is_similar, score = compute_text_similarity(first, second, model)
    sum += score
  return sum 

# maybe try computing the best similarity match with a given message, like take the first message, if it's most similar to the 
# fourth message then, within that rearrange the orders of the messages before and find a thread that contains the most similarity with 
# order, like maybe the first message and the third message, then the fourth message gives a greater similarity score
# when comparing threads with different messages, compute some kind of ratio so concentrated similarity is valued rather than just 
# the number of messages, weighted mean? 

def find_message_match(comp_message, all_messages, model): 
  similar_index = 0 
  max_score = 0

  for idx, message in enumerate(all_messages): 
    is_similar, score = compute_text_similarity(comp_message[3], message[3], model)
    if score > max_score and comp_message is not message: 
      max_score = score 
      similar_index = idx
  
  return (similar_index, max_score)

counter = 0 
for idx, message_chunk in enumerate(message_chunks):
  if counter == 3: 
    break 
  end_idx = 6 # take the first 6 messages 
  mid_idx = int(end_idx / 2)
  spliced = message_chunk[:end_idx]
  split1 = spliced[:mid_idx]
  split2 = spliced[mid_idx:]
  bruteforce_microthreads(split1, split2, embedder)
  counter += 1