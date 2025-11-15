# this file holds the part of the algorithm responsible for the higher-level chunking, mostly for topic shifts

from datetime import datetime
from display.display import display_validated_chunks, display_conversation_sections
from chunk_algorithm.similarity import compute_burst_similarity, compute_text_similarity

MAX_SECION_TIME = 30  # max amount of time two texts must stay within
LINE_BURST_TIME = 5   # max amount of time for line bursts

def compute_topic_shifts(messages=list, model=None): # pass in embedding model for similarity
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
