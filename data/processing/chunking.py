from sentence_transformers import SentenceTransformer, SimilarityFunction
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datetime import datetime
from test_data import message_chunks
from display import display_validated_chunks, display_conversation_sections
from similarity import compute_burst_similarity, compute_text_similarity, THREAD_TEXT_SIMILARITY_THRESHOLD, compute_thread_similarity
import numpy as np

MAX_SECION_TIME = 30 # max amount of time two texts must stay within
LINE_BURST_TIME = 5 # max amount of time for line bursts

def chunk_messages(messages=list, model=None): # pass in embedding model for similarity 
  # list in the format :
  # ('2024-10-08 01:31:13', '', '', 'OOUU it's so fun seeing how other people edit stuff', '+19365539666')
  all_chunks = []
  bursts = []
  line_chunks = []
  current_lines = []
  idx = 0 

  text_total = 0 
  similar_texts = 0

  burst_total = 0
  b_similar_counter = 0

  while idx < len(messages) - 1: 
    # Parse current and next timestamps - FIX: use item[0] not item[idx][0]
    current_timestamp_str = messages[idx][0] # '2024-10-08 10:00:00'
    next_timestamp_str = messages[idx + 1][0]  # Access next message from messages list

    # Convert to datetime objects for proper time calculation
    current_time = datetime.strptime(current_timestamp_str, '%Y-%m-%d %H:%M:%S')
    next_time = datetime.strptime(next_timestamp_str, '%Y-%m-%d %H:%M:%S')

    # Calculate time difference in minutes (e.g., 1:31 to 2:01 = 30 minutes)
    time_diff_minutes = (next_time - current_time).total_seconds() / 60

    current_lines.append(messages[idx])

    if time_diff_minutes < LINE_BURST_TIME: # next message  within 5 minutes 
      pass

    elif time_diff_minutes < MAX_SECION_TIME and time_diff_minutes > LINE_BURST_TIME: # between 5 and 30 minutes 
      bursts.append(current_lines)
      current_lines = [] # reset 
    
    elif time_diff_minutes > MAX_SECION_TIME: 
      is_similar, score = compute_text_similarity(messages[idx][3], messages[idx + 1][3], model)
      if is_similar is True: # keep in same section if messages are similar 
        bursts.append(current_lines)
        current_lines = []
        similar_texts += 1 # keep track of similar number of texts for metrics 
        pass
      else: # otherwise, separate sections 
        bursts.append(current_lines)
        line_chunks.append(bursts) 
        bursts = []
        current_lines = []

    idx += 1

  # Cleanup: add the last message and any remaining bursts/sections
  current_lines.append(messages[idx])  # Add the last message
  if current_lines:  # If there are lines in the current burst
    bursts.append(current_lines)
  if bursts:  # If there are bursts in the current section
    line_chunks.append(bursts)

  all_chunks = line_chunks

  # Display conversation sections and calculate metrics
  burst_total, text_total = display_conversation_sections(all_chunks)

  # Now check embedding similarity for adjacent bursts 
  # this is primarily used to mark topic shifts
  validated_chunks = []
  for chunk in all_chunks:
    # If section has only 1 burst, add it directly (no comparison needed)
    if len(chunk) == 1:
      validated_chunks.append(chunk)
      continue

    similar_bursts = []
    for idx, burst in enumerate(chunk):
      if idx == len(chunk) - 1:  # Last burst - add it if not already processed
        break
      is_similar = compute_burst_similarity(chunk[idx], chunk[idx + 1], model, idx + 1, idx + 2)
      if is_similar is True:
        similar_bursts.append((chunk[idx], chunk[idx + 1]))  # add as merged chunk
        b_similar_counter += 1 # keep track of similar bursts for metrics
      else:
        similar_bursts.append(chunk[idx])  # add individually
        similar_bursts.append(chunk[idx + 1])
    validated_chunks.append(similar_bursts)
  
  # Display validated chunks structure
  display_validated_chunks(validated_chunks)

  return validated_chunks, text_total, similar_texts, burst_total, b_similar_counter


# given two bursts, return the microthreads associated with each burst
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
    current_score = 0
    message_text = message_tuple[3]  # Extract text for comparison

    for x, comp_message_tuple in enumerate(burst2):
      comp_message_text = comp_message_tuple[3]  # Extract text for comparison
      is_similar, score = compute_text_similarity(message_text, comp_message_text, model, THREAD_TEXT_SIMILARITY_THRESHOLD)
      if is_similar is True:  # But only add to thread if above threshold
        if message_text not in seen_similar:
          seen_similar.add(message_text)
        current_score += score # only score if similar 
        micro_thread.append(comp_message_tuple)  # Store full tuple
        seen_similar.add(comp_message_text)
    if len(micro_thread) > 0: # don't add
      micro_threads.append(micro_thread)
      message_scores.append({
        "score": current_score,
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

  print("\n=== MICROTHREADS ===")
  for time_stamp, thread in valid_threads.items():
    print(f"\nThread {counter} (starts at {time_stamp}):")
    for message_tuple in thread:
      print(f"  {message_tuple[3]}")  # Print message text
    counter += 1

  if remainders:
    print("\n=== REMAINDERS ===")
    for message_tuple in remainders:
      print(f"  {message_tuple[3]}")  # Print message text
  
  # deal with remainders

def append_remainders(valid_threads, remainders, model, decay_rate): # 0.05
  
  thread_embs = [] # stores embedding of each message in each thread, 2d array 
  remainder_embs = []
  # Encode all threads

  for thread_timestamp, thread in valid_threads: # emb each message in each thread 
    batch = []
    for message in thread: 
      thread_text = " ".join([msg[3] for msg in thread])
      emb = model.encode(thread_text, convert_to_tensor=True)
      batch.append(emb)
    thread_embs.append(batch)

  # encode all remainders
  for remainder in remainders:
    r_emb = model.encode(remainder[3], convert_to_tensor=True)
    remainder_embs.append(r_emb)

  # process each remainder, keep track of scores 
  cosine_thread_scores = [] # stores an array of cosine scores for each different thread 
  for idx, remainder in enumerate(remainders):
    # compute time_diff between remainder and all micro-threads, fix later to only compare against up to 2 threads
    # and start at a certain indexing to reduce searching 
    times = [] 
    r_emb = remainder_embs[idx]
    remainder_time = datetime.strptime(remainder[0], '%Y-%m-%d %H:%M:%S')

    # Calculate time differences for all threads
    for x, thread in enumerate(valid_threads): # change this later to start at a certain index, and then search for most relevant time boundaries
      
      for idx, message in enumerate(thread): 
        thread_time = datetime.strptime(message[idx][0], '%Y-%m-%d %H:%M:%S')
        time_diff_minutes = abs((remainder_time - thread_time).total_seconds() / 60)
        times.append(time_diff_minutes)

    sims = F.cosine_similarity(r_emb.unsqueeze(0), thread_embs_stacked)
    print(sims.shape)

    decays = np.exp(-np.array(times) * decay_rate)

  print("")


embedder = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.COSINE)

# burst1 – initiating discussion about a team presentation
# burst1 – mixed context: work update + casual event mention
burst1 = [
    ('2024-11-04 09:12:00', '', '', "Morning! Did you upload the final report draft yet?", '+15551234567'),
    ('2024-11-04 09:12:45', '', '', "Also, are we still on for coffee later?", '+15551234567'),
    ('2024-11-04 09:13:20', '', '', "I added the budget section to the spreadsheet last night.", '+15551234567'),
    ('2024-11-04 09:13:55', '', '', "Need your feedback before I submit it to the client.", '+15551234567'),
]

# burst2 – replies that weave both threads together again (project + social)
burst2 = [
    ('2024-11-04 09:21:00', 'me', '', "Yeah, I just uploaded the report this morning.", '+19365539666'),
    ('2024-11-04 09:21:40', 'me', '', "The budget section looks solid, maybe expand the notes under Q3.", '+19365539666'),
    ('2024-11-04 09:22:15', 'me', '', "And yes, coffee later still works — same café as last time?", '+19365539666'),
    ('2024-11-04 09:22:45', 'me', '', "I'll bring my laptop in case we need to fix anything before submission.", '+19365539666'),
]



compute_microthreads(burst1, burst2, embedder)
# for message_chunk in message_chunks:
  # v, text_total, similar_texts, burst_total, b_similar_counter = chunk_messages(message_chunk, embedder)
  # print(f"Total texts : {text_total}, Similar texts : {similar_texts}\n")
  # print(f"Total bursts : {burst_total}, Similar bursts : {b_similar_counter}")