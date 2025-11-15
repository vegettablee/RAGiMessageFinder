# this file orchestrates the algorithm responsible for computing micro-threads between bursts

from chunk_algorithm.similarity import compute_text_similarity, THREAD_TEXT_SIMILARITY_THRESHOLD

def compute_microthreads(burst1, burst2, model):
  # burst1 and burst2 are lists of tuples: (timestamp, sender, '', text, phone)

  micro_threads = [] # keep track of all micro_threads
  seen_similar = set() # contains seen TEXT that are similar (tracking by text string)
  remainders = [] # unused messages that were not similar
  message_scores = []  # list of dictionaries, has form :
  # [{"thread" : [Message1, Message2], "score" : 0.88}, ...]
  # for every message in this thread that is similar, score keeps a running total
  # this is specifically for dealing with messages that use duplicate threads, threads with the higher score get used first
  # then, with the outlier messages, concenate them after

  # compute threads with respective scores, not checking if other threads have duplicates
  for idx, message_tuple in enumerate(burst1):
    micro_thread = []
    micro_thread.append(message_tuple)  # Store full tuple
    similarity_scores = []  # Track individual scores for mean + max calculation
    message_text = message_tuple[3]  # Extract text for comparison

    for x, comp_message_tuple in enumerate(burst2):
      comp_message_text = comp_message_tuple[3]  # Extract text for comparison
      is_similar, score = compute_text_similarity(message_text, comp_message_text, model, THREAD_TEXT_SIMILARITY_THRESHOLD)
      if is_similar is True:  # only add to thread if above threshold
        if message_text not in seen_similar:
          seen_similar.add(message_text)
        similarity_scores.append(score)  # Track individual scores
        micro_thread.append(comp_message_tuple)  # Store full tuple
        seen_similar.add(comp_message_text)

    # calculate score using a mean + max approach, to make the algorithm less greedy
    if len(similarity_scores) > 0:
      mean_score = sum(similarity_scores) / len(similarity_scores)
      max_score = max(similarity_scores)
      alpha = 0.5  # Weight for mean
      beta = 0.5   # Weight for max
      combined_score = alpha * mean_score + beta * max_score
    else:
      combined_score = 0

    if len(micro_thread) > 0: # don't add
      micro_threads.append(micro_thread)
      message_scores.append({
        "score": combined_score,
        "thread": micro_thread,
      })

  for message_tuple in burst1:
    if message_tuple[3] not in seen_similar:  # Check text
      remainders.append(message_tuple)

  for message_tuple in burst2:
    if message_tuple[3] not in seen_similar:  # Check text
      remainders.append(message_tuple)

  sorted_list = sorted(message_scores, key=lambda x: x["score"], reverse=True) # highest to lowest scores

  seen = set()  # Track seen message TEXT to deduplicate
  valid_threads = {}
  # time_stamp : thread, where time_stamp is a key

  for idx in range(len(sorted_list)):
    thread = sorted_list[idx]["thread"]  # thread is list of tuples
    filtered_thread = []
    for message_tuple in thread:
      message_text = message_tuple[3]  # Extract text
      if message_text not in seen:  # only add unseen messages
        filtered_thread.append(message_tuple)
        seen.add(message_text)
    if len(filtered_thread) == 1:  # only add thread if it has 1 message
      remainders.append(filtered_thread[0])  # Append the tuple, not list
    elif len(filtered_thread) > 1:
      time_stamp = filtered_thread[0][0]  # extract time stamp, TO DO: only include the date
      valid_threads[time_stamp] = filtered_thread

  counter = 0

  print("\n MICROTHREADS")
  for time_stamp, thread in valid_threads.items():
    print(f"\nThread {counter} (starts at {time_stamp}):")
    for message_tuple in thread:
      print(f"  {message_tuple[3]}")  # Print message text
    counter += 1

  if remainders:
    print("\nREMAINDERS")
    for message_tuple in remainders:
      print(f"  {message_tuple[3]}")  # Print message text

  return valid_threads, remainders
  # deal with remainders

