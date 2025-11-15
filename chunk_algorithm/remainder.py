# this file handles remainder processing for microthreads

from datetime import datetime
import torch
import math
from chunk_algorithm.similarity import compute_text_similarity

def append_remainders(valid_threads, remainders, model): # 0.05

  thread_embs = [] # shape of [[(message_tuple, embedding)]], 2d array
  remainder_embs = [] # shape of [(message_tuple, embedding)]
  # Encode all threads

  for thread_timestamp, thread in valid_threads.items(): # emb each message in each thread
    batch = []
    for message in thread:
      thread_text = " ".join([msg[3] for msg in thread])
      emb = model.encode(thread_text, convert_to_tensor=True)
      tuple = (message, emb)
      batch.append(tuple)
    thread_embs.append(batch)

  # encode all remainders
  for remainder in remainders:
    r_emb = model.encode(remainder[3], convert_to_tensor=True)
    tuple = (remainder, r_emb)
    remainder_embs.append(tuple)

  # process each remainder, keep track of scores
  all_thread_scores = [] # same shape as thread_scores
  # stores an array of cosine scores for each different thread, sum this after for comparison

  for idx, remainder in enumerate(remainder_embs):
    # compute time_diff between remainder and all micro-threads, fix later to only compare against up to 2 threads
    # and start at a certain indexing to reduce searching
    r_emb = remainder_embs[idx][1]
    remainder_time = datetime.strptime(remainder[0][0], '%Y-%m-%d %H:%M:%S')

    remainder_scores = []  # scores for this remainder against all threads
    # Calculate time differences for all threads
    for x, thread in enumerate(thread_embs): # change this later to start at a certain index, and then search for most relevant time boundaries
      batch_scores = [] # contains the scores from the remainder to all of the messages from the thread, gets added to all_threads_scores
      for m_idx, message_tuple in enumerate(thread):
        thread_time = datetime.strptime(message_tuple[0][0], '%Y-%m-%d %H:%M:%S')
        time_diff_minutes = abs((remainder_time - thread_time).total_seconds() / 60)
        decay_multiplier = time_decay(time_diff_minutes)
        # compute score between remainder and thread score
        is_similar, score = compute_text_similarity(message_tuple[0][3], remainder[0][3], model)
        final_score = score * decay_multiplier
        print("Decay score : " + str(final_score))
        batch_scores.append(final_score)
      remainder_scores.append(batch_scores)
    all_thread_scores.append(remainder_scores)

  threads_list = list(valid_threads.values())
  seen_indices = {}
  for r_idx, remainder in enumerate(remainders):
    remainder_thread_scores = all_thread_scores[r_idx]  # get scores for this remainder against all threads
    current_max_rem_index = 0
    current_max_score = 0
    current_max_thread_idx = 0
    for t_idx, thread_scores in enumerate(remainder_thread_scores):
      score_tensor = torch.tensor(thread_scores)
      max_value = torch.max(score_tensor)
      max_index = torch.argmax(score_tensor)
      if max_value > current_max_score:
        current_max_rem_index = r_idx
        current_max_score = max_value
        current_max_thread_idx = t_idx
      print("Remainder : " + remainder[3] + '\n' + "Max score : " + str(max_value.item()) + " at thread " + str(t_idx) + ", message index " + str(max_index.item()))
      print("Complement : " + threads_list[t_idx][max_index][3] + "\n\n")
    value = seen_indices.get(current_max_thread_idx, set()) # return an empty set if thread doesn't exist inside of the dictionary
    if len(value) == 0:
      exists = False
    else :
      exists = True
    if exists is True: # thread already exists in dictionary, add remainder index
      seen_indices[current_max_thread_idx].add(current_max_rem_index)
    else: # thread doesn't exist in dictionary
      seen_indices.update({
      current_max_thread_idx : set([current_max_rem_index])
      })

  # now, add all of the remainders back in chronological order with the messages
  final_threads = []
  threads_with_rem = set([key for key in seen_indices.keys()]) # thread indexes with remainders

  for thread_idx, thread in enumerate(threads_list):
    if thread_idx not in threads_with_rem: # thread has no remainders, append like normal
      final_threads.append(thread)
    else:
      rem_indices = list(seen_indices[thread_idx])
      rems = [remainders[i] for i in rem_indices]
      modified_thread = add_remainders_to_thread(rems, threads_list[thread_idx])
      final_threads.append(modified_thread)

  print("\n=== FINAL THREADS WITH REMAINDERS ===")
  for t_idx, thread in enumerate(final_threads):
    print(f"\n--- Thread {t_idx} ---")
    for msg_idx, message in enumerate(thread):
      timestamp = message[0]
      text = message[3]
      # Check if this message is a remainder
      is_remainder = message in remainders
      marker = "[REMAINDER]" if is_remainder else "[ORIGINAL] "
      print(f"{msg_idx}: {marker} {timestamp} - {text}")

  return final_threads

# time decay multiplier, starts 0.75 after 3 days, and 0.5 after a week and so on, caps at out 0.2
# this is to prevent further time stamps from making a difference unless they are directly related

def add_remainders_to_thread(remainders, thread): # go through the thread, and add the remainder to the corresponding index based on the time
  # Deduplicate remainders by text before inserting
  print(f"[DEBUG add_remainders_to_thread] Input: {len(remainders)} remainders, {len(thread)} messages in thread")

  seen_texts = set()
  unique_remainders = []
  for remainder in remainders:
    remainder_text = remainder[3]
    if remainder_text not in seen_texts:
      unique_remainders.append(remainder)
      seen_texts.add(remainder_text)

  print(f"[DEBUG add_remainders_to_thread] After dedup: {len(unique_remainders)} unique remainders")

  for remainder in unique_remainders:
    remainder_time = datetime.strptime(remainder[0], '%Y-%m-%d %H:%M:%S')
    inserted = False
    for idx, message in enumerate(thread):
      message_time = datetime.strptime(message[0], '%Y-%m-%d %H:%M:%S')
      if remainder_time < message_time:
        # insert remainder before this message
        thread = thread[:idx] + [remainder] + thread[idx:]
        inserted = True
        break  # IMPORTANT: break immediately after insertion

    # If not inserted yet, add at the end (all messages were before the remainder)
    if not inserted:
      thread = thread + [remainder]

  print(f"[DEBUG add_remainders_to_thread] Output: thread now has {len(thread)} messages\n")
  return thread


def time_decay(minutes_diff: float) -> float:
  hours = minutes_diff / 60.0
  k = 0.0114      # steepness
  t_mid = 168.0   # midpoint (hours) = 1 week
  return 1 / (1 + math.exp(k * (hours - t_mid)))
