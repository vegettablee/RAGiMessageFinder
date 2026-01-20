from backend.data.processing import file
from backend.data.processing.message_loader import getMessageCountBySubject, getMessagesBySubject
from backend.rag.pipeline.classes import DataState, NormalizedMessage, Thread, VectorChunk, TimeAnalytics


def load_raw_messages(subject_phone : str, num_messages : int) -> list[tuple]: 
    messages = getMessagesBySubject(subject_phone, num_messages)
    return messages

def find_max_messages(subject_phone) -> int: 
    num_messages = getMessageCountBySubject(subject_phone)
    print(f"number of messages found with {subject_phone}: ", num_messages)
    return num_messages

def normalize_raw_messages(raw_messages : list) -> list[NormalizedMessage]:
    """
    Convert raw message tuples to NormalizedMessage objects.

    Raw message tuple format (from message_loader.getMessagesBySubject):
      (sent_at, sender_label, '', body, subject_out)
      - sent_at: timestamp string like "2024-10-08 01:31:13"
      - sender_label: "me" or "" (empty for incoming)
      - '': legacy empty field
      - body: message content text
      - subject_out: normalized phone number

    Args:
        raw_messages: List of raw message tuples

    Returns:
        List of NormalizedMessage objects with order_index and parsed timestamps
    """
    from datetime import datetime

    normalized = []

    for idx, msg_tuple in enumerate(raw_messages):
        sent_at, sender_label, _, body, phone = msg_tuple

        # Parse timestamp string to datetime object
        timestamp = datetime.strptime(sent_at, "%Y-%m-%d %H:%M:%S")

        # Generate unique message_id using index and timestamp
        message_id = f"{phone}_{idx}_{timestamp.strftime('%Y%m%d%H%M%S')}"

        # Create NormalizedMessage instance
        normalized_msg = NormalizedMessage(
            message_id=message_id,
            contact_id=phone,
            sender=sender_label if sender_label else "them",  # "me" or "them"
            timestamp=timestamp,
            order_index=idx,
            content=body
        )

        normalized.append(normalized_msg)

    return normalized

def create_message_lookup(chunks: list) -> dict[str, dict]:
    """
    Create a lookup dictionary mapping message_id to message content.

    Args:
        chunks: List of message chunks, where each chunk is a list of raw message tuples
                Tuple format: (timestamp, sender_label, '', text, phone)

    Returns:
        Dictionary mapping message_id -> {sender, text, timestamp}
    """
    from datetime import datetime

    message_lookup = {}

    for chunk_idx, chunk in enumerate(chunks):
        for idx, msg in enumerate(chunk):
            # Parse tuple: (timestamp, sender_label, '', text, phone)
            timestamp_str = msg[0]
            sender_label = msg[1]
            text = msg[3]
            phone = msg[4]

            # Create message_id matching the format in normalize_chunks
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            message_id = f"{phone}_{chunk_idx}_{idx}_{timestamp.strftime('%Y%m%d%H%M%S')}"

            # Store message content
            message_lookup[message_id] = {
                'sender': sender_label if sender_label else "them",
                'text': text,
                'timestamp': timestamp_str
            }

    return message_lookup

def normalize_chunks(chunks : list) -> list[Thread]:
    """
    Convert chunked messages into Thread objects.

    Args:
        chunks: List of message chunks, where each chunk is a list of raw message tuples
                Tuple format: (timestamp, sender_label, '', text, phone)

    Returns:
        List of Thread objects with metadata extracted from messages
    """
    from datetime import datetime

    threads = []

    for chunk_idx, chunk in enumerate(chunks):
        # get the meta-data based on the first message and the last
        first_msg = chunk[0]
        last_msg = chunk[-1]

        # Parse tuple elements: (timestamp, sender_label, '', text, phone)
        first_timestamp = datetime.strptime(first_msg[0], "%Y-%m-%d %H:%M:%S")
        last_timestamp = datetime.strptime(last_msg[0], "%Y-%m-%d %H:%M:%S")
        phone = first_msg[4]

        # Create message_ids for all messages in chunk
        message_ids = [
            f"{msg[4]}_{chunk_idx}_{idx}_{datetime.strptime(msg[0], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')}"
            for idx, msg in enumerate(chunk)
        ]

        # Create unique thread_id using chunk index and first message timestamp
        thread_id = f"thread_{chunk_idx}_{first_timestamp.strftime('%Y%m%d%H%M%S')}"

        # Get start and end times from first and last messages
        start_time = first_timestamp
        end_time = last_timestamp

        # Create metadata dictionary with useful thread information
        metadata = {
            'message_count': len(chunk),
            'contact_id': phone,
            'duration_seconds': (end_time - start_time).total_seconds(),
            'chunk_index': chunk_idx
        }

        # Create Thread object
        thread = Thread(
            thread_id=thread_id,
            message_ids=message_ids,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata
        )

        threads.append(thread)

    return threads 