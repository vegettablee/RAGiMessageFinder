from sentence_transformers import SentenceTransformer, SimilarityFunction
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datetime import datetime
from data.processing.test_data import message_chunks
from display.display import display_validated_chunks, display_conversation_sections
from algorithm.similarity import compute_burst_similarity, compute_text_similarity, THREAD_TEXT_SIMILARITY_THRESHOLD
import numpy as np
import math 

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
  append_remainders(valid_threads, remainders, model)
  # deal with remainders

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

# time decay multiplier, starts 0.75 after 3 days, and 0.5 after a week and so on, caps at out 0.2 
# this is to prevent further time stamps from making a difference unless they are directly related 
def time_decay(minutes_diff: float) -> float:
  hours = minutes_diff / 60.0
  k = 0.0114      # steepness
  t_mid = 168.0   # midpoint (hours) = 1 week
  return 1 / (1 + math.exp(k * (hours - t_mid)))

def add_remainders_to_thread(remainders, thread): # go through the thread, and add the remainder to the corresponding index based on the time
  for remainder in remainders: 
    remainder_time = datetime.strptime(remainder[0], '%Y-%m-%d %H:%M:%S')
    inserted = False 
    for idx, message in enumerate(thread):
      message_time = datetime.strptime(message[0], '%Y-%m-%d %H:%M:%S')
      if remainder_time < message_time:
      # insert remainder before this message 
        thread = thread[:idx] + [remainder] + thread[idx:]
        inserted = True 
      if inserted is True: # remainder inserted already, break out of thread and move to next remainder 
        break

  # all messages were before the remainder, add remainder to the end 
  return thread
  

embedder = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.COSINE)

# burst1 – initiating discussion about a team presentation
# burst1 – mixed context: work update + casual event mention
burst1_tc1 = [
    ('2024-11-04 14:05:00', '', '', "Hey! Did you book the hotel for the Austin trip?", '+15551234567'),
    ('2024-11-04 14:05:30', '', '', "Also the Jenkins presentation is due Friday right?", '+15551234567'),
    ('2024-11-04 14:06:15', '', '', "I was thinking we could check out that BBQ place everyone talks about.", '+15551234567'),
    ('2024-11-04 14:06:50', '', '', "Can you send me the slides template you used last quarter?", '+15551234567'),
    ('2024-11-04 14:07:20', '', '', "Oh and my sister wanted to know if you still have those concert tickets.", '+15551234567'),
]

burst2_tc1 = [
    ('2024-11-04 14:15:00', 'me', '', "Hotel's booked! Got us rooms at the Hilton downtown.", '+19365539666'),
    ('2024-11-04 14:15:35', 'me', '', "Jenkins presentation is actually Thursday, not Friday.", '+19365539666'),
    ('2024-11-04 14:16:10', 'me', '', "Franklin BBQ? We'd need to get there early, the line gets crazy.", '+19365539666'),
    ('2024-11-04 14:16:45', 'me', '', "I'll email you the template when I get back to my desk.", '+19365539666'),
    ('2024-11-04 14:17:20', 'me', '', "Yeah I still have the tickets, she can pick them up this weekend.", '+19365539666'),
]

# Test Case 2: Single topic that appears fragmented (all about one project)
burst1_tc2 = [
    ('2024-11-04 10:30:00', '', '', "The API integration is throwing errors again.", '+15559876543'),
    ('2024-11-04 10:30:45', '', '', "Also Karen from QA wants access to the staging environment.", '+15559876543'),
    ('2024-11-04 10:31:30', '', '', "I think it's the same authentication issue from last week.", '+15559876543'),
    ('2024-11-04 10:32:00', '', '', "Should we roll back to the previous version?", '+15559876543'),
]

burst2_tc2 = [
    ('2024-11-04 10:40:00', 'me', '', "Let me check the logs first before we roll back.", '+19365539666'),
    ('2024-11-04 10:40:30', 'me', '', "I'll give Karen staging access, same permissions as last time?", '+19365539666'),
    ('2024-11-04 10:41:15', 'me', '', "You're right, it's the OAuth token expiration thing again.", '+19365539666'),
    ('2024-11-04 10:41:50', 'me', '', "I can patch it in about an hour, no need to roll back.", '+19365539666'),
]

# Test Case 3: Rapid topic switching with minimal context
burst1_tc3 = [
    ('2024-11-04 16:20:00', '', '', "Dinner tonight?", '+15552468101'),
    ('2024-11-04 16:20:15', '', '', "Your package arrived at my place", '+15552468101'),
    ('2024-11-04 16:20:40', '', "Italian or Thai?", '+15552468101'),
    ('2024-11-04 16:21:00', '', '', "The big brown box", '+15552468101'),
    ('2024-11-04 16:21:25', '', '', "Maybe around 7?", '+15552468101'),
]

burst2_tc3 = [
    ('2024-11-04 16:28:00', 'me', '', "Thai sounds great!", '+19365539666'),
    ('2024-11-04 16:28:20', 'me', '', "Oh perfect, I'll grab the package tomorrow morning", '+19365539666'),
    ('2024-11-04 16:28:50', 'me', '', "7 works, meet at the restaurant?", '+19365539666'),
]

# Test Case 4: Topic revival (topic mentioned, dropped, then returned to)
burst1_tc4 = [
    ('2024-11-04 11:00:00', '', '', "Did you see the budget numbers for Q4?", '+15553691470'),
    ('2024-11-04 11:00:40', '', '', "They're way higher than I expected.", '+15553691470'),
    ('2024-11-04 11:01:20', '', '', "Btw are you coming to Sarah's birthday thing Saturday?", '+15553691470'),
    ('2024-11-04 11:02:00', '', '', "She's doing a potluck at her place.", '+15553691470'),
    ('2024-11-04 11:03:15', '', '', "Back to the budget - did marketing really spend that much on ads?", '+15553691470'),
    ('2024-11-04 11:03:50', '', '', "Seems like we should schedule a review meeting.", '+15553691470'),
]

burst2_tc4 = [
    ('2024-11-04 11:12:00', 'me', '', "Yeah the Q4 numbers are wild, marketing went over by like 30%.", '+19365539666'),
    ('2024-11-04 11:12:35', 'me', '', "I can make it to Sarah's! I'll bring a salad or something.", '+19365539666'),
    ('2024-11-04 11:13:10', 'me', '', "Let's do the budget review Wednesday afternoon?", '+19365539666'),
    ('2024-11-04 11:13:40', 'me', '', "I'll send a calendar invite and pull the detailed reports.", '+19365539666'),
]

# Test Case 5: Ambiguous references (testing pronoun/reference resolution)
burst1_tc5 = [
    ('2024-11-04 13:45:00', '', '', "Can you review it before end of day?", '+15558675309'),
    ('2024-11-04 13:45:30', '', '', "Also did you hear back from them?", '+15558675309'),
    ('2024-11-04 13:46:10', '', '', "The contract needs those changes we discussed.", '+15558675309'),
    ('2024-11-04 13:46:45', '', '', "And the vendor said they'd call by 2pm.", '+15558675309'),
]

burst2_tc5 = [
    ('2024-11-04 13:55:00', 'me', '', "I'll review the contract by 4pm.", '+19365539666'),
    ('2024-11-04 13:55:30', 'me', '', "No response from the vendor yet, it's almost 2 though.", '+19365539666'),
    ('2024-11-04 13:56:00', 'me', '', "What specific changes did we want in the contract again?", '+19365539666'),
]

# Test Case 6: Time-based topic clustering (morning routine vs evening plans)
burst1_tc6 = [
    ('2024-11-04 08:15:00', '', '', "Morning! Don't forget the 9am standup.", '+15554823691'),
    ('2024-11-04 08:15:30', '', '', "I grabbed you a coffee, at your desk.", '+15554823691'),
    ('2024-11-04 08:16:15', '', '', "For tonight - should we do drinks after work or just head to the game?", '+15554823691'),
    ('2024-11-04 08:16:50', '', '', "Oh and bring the design mockups to standup.", '+15554823691'),
    ('2024-11-04 08:17:20', '', '', "Game starts at 7:30 so we'd need to leave by 6.", '+15554823691'),
]

burst2_tc6 = [
    ('2024-11-04 08:25:00', 'me', '', "Thanks for the coffee! I have the mockups ready.", '+19365539666'),
    ('2024-11-04 08:25:30', 'me', '', "I'll be at standup in 5 minutes.", '+19365539666'),
    ('2024-11-04 08:26:00', 'me', '', "Let's skip drinks and go straight to the game, better seats that way.", '+19365539666'),
    ('2024-11-04 08:26:30', 'me', '', "I can drive if you want.", '+19365539666'),
]

# Test Case 7: Nested sub-topics (main project with multiple components)
burst1_tc7 = [
    ('2024-11-04 15:00:00', '', '', "The database migration is scheduled for midnight.", '+15557418529'),
    ('2024-11-04 15:00:45', '', '', "Did you update the user-facing announcement?", '+15557418529'),
    ('2024-11-04 15:01:30', '', '', "We need to back up everything before we start.", '+15557418529'),
    ('2024-11-04 15:02:10', '', '', "Also the customer support team needs talking points.", '+15557418529'),
    ('2024-11-04 15:02:50', '', '', "I'll handle the technical rollback plan.", '+15557418529'),
    ('2024-11-04 15:03:30', '', '', "Make sure the announcement mentions the 2-hour maintenance window.", '+15557418529'),
]

burst2_tc7 = [
    ('2024-11-04 15:12:00', 'me', '', "Announcement is drafted, sending it to you now for review.", '+19365539666'),
    ('2024-11-04 15:12:40', 'me', '', "I'll start the backup process at 11pm.", '+19365539666'),
    ('2024-11-04 15:13:15', 'me', '', "The support talking points doc is in the shared folder.", '+19365539666'),
    ('2024-11-04 15:13:50', 'me', '', "I included the maintenance window timing in the announcement.", '+19365539666'),
    ('2024-11-04 15:14:25', 'me', '', "Let me know if the rollback plan needs anything else.", '+19365539666'),
]

burst1_tc8 = [
    ('2024-11-04 12:00:00', '', '', "The client called about the redesign.", '+15551237890'),
    ('2024-11-04 12:00:35', '', '', "They want to add a blog section.", '+15551237890'),
    ('2024-11-04 12:01:15', '', '', "Also they're concerned about the timeline.", '+15551237890'),
    ('2024-11-04 12:01:50', '', '', "The blog needs to be done before the main launch.", '+15551237890'),
    ('2024-11-04 12:02:30', '', '', "Can we push the deadline back two weeks?", '+15551237890'),
    ('2024-11-04 12:03:10', '', '', "Or should we phase the blog as v2?", '+15551237890'),
]

burst2_tc8 = [
    ('2024-11-04 12:12:00', 'me', '', "I think phasing makes more sense.", '+19365539666'),
    ('2024-11-04 12:12:35', 'me', '', "We can launch the redesign on time without the blog.", '+19365539666'),
    ('2024-11-04 12:13:10', 'me', '', "Then roll out the blog section two weeks later.", '+19365539666'),
    ('2024-11-04 12:13:45', 'me', '', "That way the timeline stays intact for the main deliverable.", '+19365539666'),
]




compute_microthreads(burst1_tc8, burst2_tc8, embedder)
# for message_chunk in message_chunks:
  # v, text_total, similar_texts, burst_total, b_similar_counter = chunk_messages(message_chunk, embedder)
  # print(f"Total texts : {text_total}, Similar texts : {similar_texts}\n")
  # print(f"Total bursts : {burst_total}, Similar bursts : {b_similar_counter}")