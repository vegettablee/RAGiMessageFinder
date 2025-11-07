from sentence_transformers import SentenceTransformer, SimilarityFunction
from data.processing.test_data import message_chunks
from algorithm.topic_shift import compute_topic_shifts
from algorithm.micro_thread import compute_microthreads
from algorithm.remainder import append_remainders
from algorithm.similarity import compute_text_similarity, THREAD_TEXT_SIMILARITY_THRESHOLD
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
  
  pair = find_message_match(messages[0], messages, model)
  start_idx = 0
  similar_idx = pair[0]
  pair_score = pair[1]

  order = find_optimal_order(messages, start_idx, similar_idx, model)

  return max_threads, max_sum

def find_optimal_order(all_messages, start_idx, end_idx, model): 
  max_permutations = math.pow(2, end_idx - start_idx) - 1
  optimal_order = []
  highest_score = 0 
  global_score = 0 
  used = []
  for i in range(start_idx, end_idx):
    sequence = all_messages[start_idx:end_idx]
    threads = subsequences(sequence, model=model)
    highest_score_idx = torch.argsort([pair[1] for pair in threads])

    best_current_thread = threads[highest_score_idx][0]
    best_current_score = threads[highest_score_idx][1]
    used.append({
      "micro_thread" : best_current_thread, 
      "score" : best_current_score
    })
    print(f"Best micro-thread for {start_idx} - {end_idx}\n")
    print(str(best_current_thread))
    print("Score : " + str(best_current_score))
    start_idx += 1
      

# helper function for find_optimal_order, computes all non-contiguous subsequences with length >= 2
def subsequences(seq, start=0, current=None, model=None):
  if current is None:
    current = []

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
  if counter == 1: 
    break 
  mid_idx = int(len(message_chunk) / 2)
  bruteforce_microthreads(message_chunk[mid_idx:], message_chunk[:mid_idx], embedder)
  counter += 1