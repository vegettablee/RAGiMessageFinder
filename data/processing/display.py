from datetime import datetime

# this file is just for displaying chunking metrics 

def format_message(msg_tuple):
  """Format a single message for display"""
  timestamp_str = msg_tuple[0]  # '2024-10-15 14:30:00'
  sender = msg_tuple[1]  # 'me' or ''
  message_text = msg_tuple[3]

  # Parse timestamp to get month and day
  dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
  date_str = dt.strftime('%B %d')  # e.g., "October 15"

  # Format based on sender
  if sender == 'me':
    return f"Me : {message_text}        Sent on {date_str}"
  else:
    return f"{message_text}        Sent on {date_str}"


def print_burst(burst, indent="    "):
  """Print all messages in a burst"""
  for msg in burst:
    print(f"{indent}{format_message(msg)}")


def display_conversation_sections(all_chunks):
  """Display conversation sections with bursts and calculate metrics"""
  burst_total = 0
  text_total = 0

  print(f"Total conversation sections: {len(all_chunks)}")
  for i, section in enumerate(all_chunks):
    print(f"\nSection {i + 1}: {len(section)} line bursts")
    for j, burst in enumerate(section):
      messages_text = [msg[3] for msg in burst]
      print(f"  Burst {j + 1}: {len(burst)} messages - {messages_text}")
      num_messages = len(burst)
      burst_total += 1  # keep track of total bursts for metrics
      text_total += num_messages

  return burst_total, text_total


def display_validated_chunks(validated_chunks):
  """Display the structure of validated chunks"""
  print("\n=== VALIDATED CHUNKS STRUCTURE ===")
  print(f"Total validated chunks: {len(validated_chunks)}\n")

  for i, chunk in enumerate(validated_chunks):
    print(f"\nValidated Chunk {i + 1}:")

    if not isinstance(chunk, list) or len(chunk) == 0:
      print("  EMPTY CHUNK")
      continue

    # Determine if chunk contains message tuples directly or bursts/merged bursts
    first_elem = chunk[0]

    # Check if first element is a tuple (could be merged bursts or single message)
    if isinstance(first_elem, tuple):
      # Check if it's a tuple of bursts (merged) by seeing if first element is a list
      if len(first_elem) > 0 and isinstance(first_elem[0], list):
        # Merged bursts: chunk = [(burst1, burst2)]
        print(f"  MERGED BURSTS:")
        for burst_idx, burst in enumerate(first_elem):
          print(f"    --- Burst {burst_idx + 1} ---")
          print_burst(burst, "    ")
      else:
        # Single message tuple (edge case) - treat chunk as single burst
        print(f"  SINGLE BURST:")
        print_burst(chunk, "    ")

    # Check if first element is a list (burst)
    elif isinstance(first_elem, list):
      # List of bursts
      if len(chunk) == 1:
        print(f"  SINGLE BURST:")
        print_burst(first_elem, "    ")
      else:
        print(f"  SEPARATE BURSTS:")
        for burst_idx, burst in enumerate(chunk):
          if isinstance(burst, list):
            print(f"    --- Burst {burst_idx + 1} ---")
            print_burst(burst, "    ")

    print()
