import os
import glob
import random
from collections import defaultdict

shuffle_mode = False

def load_irc_file(ascii_path, annotation_path):
  """
  Load a single IRC conversation file with its annotations.

  Args:
    ascii_path: Path to .ascii.txt file
    annotation_path: Path to .annotation.txt file

  Returns:
    dict with keys: 'id', 'raw', 'date', 'connections'
  """
  # Read messages
  with open(ascii_path, 'r', encoding='utf-8', errors='ignore') as f:
    messages = [line.strip() for line in f.readlines()]

  # Read annotations - build connections dict
  connections_dict = defaultdict(list)
  with open(annotation_path, 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) >= 2:
        source_id = int(parts[0])
        target_id = int(parts[1])
        connections_dict[source_id].append(target_id)

  # Only include messages that have annotations (>= 1000)
  min_id = min(connections_dict.keys()) if connections_dict else 1000
  max_id = max(connections_dict.keys()) if connections_dict else len(messages) - 1

  # Extract messages from min_id to max_id
  original_ids = list(range(min_id, max_id + 1))
  raw_messages = [messages[i] for i in original_ids if i < len(messages)]

  # Create mapping from original IDs to renumbered IDs (1, 2, 3, ...)
  id_mapping = {orig_id: idx + 1 for idx, orig_id in enumerate(original_ids[:len(raw_messages)])}

  # Renumber IDs to start from 1
  renumbered_ids = list(range(1, len(raw_messages) + 1))

  # Build connections list with renumbered IDs
  connections = []
  for orig_id in original_ids[:len(raw_messages)]:
    orig_connections = connections_dict.get(orig_id, [orig_id])
    # Convert original connection IDs to renumbered IDs
    renumbered_connections = [id_mapping.get(conn_id, id_mapping[orig_id]) for conn_id in orig_connections if conn_id in id_mapping]
    if not renumbered_connections:
      renumbered_connections = [id_mapping[orig_id]]
    connections.append(renumbered_connections)

  # Extract dates from messages (all same date for one file)
  date = 'unknown'

  return {
    'id': renumbered_ids,
    'raw': raw_messages,
    'date': [date] * len(raw_messages),
    'connections': connections
  }


def load_dataset(split='train', num_examples=None, data_dir='/Users/prestonrank/RAGMessages/dataset/irc-disentanglement/data'):
  """
  Load IRC disentanglement dataset.

  Args:
    split: 'train', 'dev', or 'test'
    num_examples: Number of conversation files to load (None = all)
    data_dir: Path to dataset directory

  Returns:
    List of conversation dictionaries
  """
  split_dir = os.path.join(data_dir, split)

  # Find all .ascii.txt files
  ascii_files = sorted(glob.glob(os.path.join(split_dir, '*.ascii.txt')))

  if num_examples:
    ascii_files = ascii_files[:num_examples]

  dataset = []
  for ascii_path in ascii_files:
    # Get corresponding annotation file
    annotation_path = ascii_path.replace('.ascii.txt', '.annotation.txt')

    if os.path.exists(annotation_path):
      try:
        example = load_irc_file(ascii_path, annotation_path)
        if len(example['raw']) > 0:  # Only add non-empty conversations
          dataset.append(example)
      except Exception as e:
        print(f"Error loading {ascii_path}: {e}")
        continue

  if shuffle_mode:
    random.shuffle(dataset)

  print(f"Loaded {len(dataset)} conversations from {split} split")
  return dataset


def get_dataloader(dataset, batch_size=1, shuffle=False):
  """
  Simple dataloader that yields batches of conversations.

  Args:
    dataset: List of conversation dicts
    batch_size: Batch size
    shuffle: Whether to shuffle

  Yields:
    Batches of conversations
  """
  if shuffle:
    random.shuffle(dataset)

  for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i + batch_size]
    yield batch


# Global dataset cache
_dataset_cache = None

def get_example():
  """Get a single random example from the training set."""
  global _dataset_cache

  if _dataset_cache is None:
    _dataset_cache = load_dataset(split='train', num_examples=10)

  return random.choice(_dataset_cache)

example = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'raw': [
        "[09:15] <alice> does anyone have experience with docker compose?",
        "[09:16] <bob> alice: yeah, what's the issue?",
        "[09:16] <charlie> morning all! quick question about git rebase",
        "[09:17] <alice> bob: my containers keep restarting, not sure why",
        "[09:17] <dave> charlie: shoot, what do you need?",
        "[09:18] <bob> alice: check your logs with docker logs <container_name>",
        "[09:18] <charlie> dave: if I rebase on main, will I lose my commits?",
        "[09:19] <alice> bob: ah it's a memory issue, thanks!",
        "[09:19] <eve> anyone free for code review?",
        "[09:20] <dave> charlie: no, rebase rewrites history but keeps your commits",
        "[09:20] <bob> eve: sure, post the PR link",
        "[09:21] <eve> bob: here you go github.com/project/pr/123"
    ],
    'ascii': [
        "[09:15] <alice> does anyone have experience with docker compose?",
        "[09:16] <bob> alice: yeah, what's the issue?",
        "[09:16] <charlie> morning all! quick question about git rebase",
        "[09:17] <alice> bob: my containers keep restarting, not sure why",
        "[09:17] <dave> charlie: shoot, what do you need?",
        "[09:18] <bob> alice: check your logs with docker logs <container_name>",
        "[09:18] <charlie> dave: if I rebase on main, will I lose my commits?",
        "[09:19] <alice> bob: ah it's a memory issue, thanks!",
        "[09:19] <eve> anyone free for code review?",
        "[09:20] <dave> charlie: no, rebase rewrites history but keeps your commits",
        "[09:20] <bob> eve: sure, post the PR link",
        "[09:21] <eve> bob: here you go github.com/project/pr/123"
    ],
    'tokenized': [
        ['09:15', 'alice', 'does', 'anyone', 'have', 'experience', 'with', 'docker', 'compose'],
        ['09:16', 'bob', 'alice', 'yeah', 'what', 's', 'the', 'issue'],
        ['09:16', 'charlie', 'morning', 'all', 'quick', 'question', 'about', 'git', 'rebase'],
        ['09:17', 'alice', 'bob', 'my', 'containers', 'keep', 'restarting', 'not', 'sure', 'why'],
        ['09:17', 'dave', 'charlie', 'shoot', 'what', 'do', 'you', 'need'],
        ['09:18', 'bob', 'alice', 'check', 'your', 'logs', 'with', 'docker', 'logs', 'container_name'],
        ['09:18', 'charlie', 'dave', 'if', 'i', 'rebase', 'on', 'main', 'will', 'i', 'lose', 'my', 'commits'],
        ['09:19', 'alice', 'bob', 'ah', 'it', 's', 'a', 'memory', 'issue', 'thanks'],
        ['09:19', 'eve', 'anyone', 'free', 'for', 'code', 'review'],
        ['09:20', 'dave', 'charlie', 'no', 'rebase', 'rewrites', 'history', 'but', 'keeps', 'your', 'commits'],
        ['09:20', 'bob', 'eve', 'sure', 'post', 'the', 'pr', 'link'],
        ['09:21', 'eve', 'bob', 'here', 'you', 'go', 'github', 'com', 'project', 'pr', '123']
    ],
    'date': ['2004-12-25', '2004-12-25', '2004-12-25', '2004-12-25', 
             '2004-12-25', '2004-12-25', '2004-12-25', '2004-12-25',
             '2004-12-25', '2004-12-25', '2004-12-25', '2004-12-25'],
    'connections': [
        [1],        # Node 1: alice starts docker question
        [1],        # Node 2: bob replies to alice (Node 1)
        [3],        # Node 3: charlie starts git question
        [2],        # Node 4: alice replies to bob (Node 2)
        [3],        # Node 5: dave replies to charlie (Node 3)
        [4],        # Node 6: bob replies to alice (Node 4)
        [5],        # Node 7: charlie replies to dave (Node 5)
        [6],        # Node 8: alice replies to bob (Node 6)
        [9],        # Node 9: eve starts code review question
        [7],        # Node 10: dave replies to charlie (Node 7)
        [9],        # Node 11: bob replies to eve (Node 9)
        [11]        # Node 12: eve replies to bob (Node 11)
    ]
}