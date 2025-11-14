display_counter = True # displays counter associated with each node 

def build_message_tree(group):

  raw_messages = group["raw"]
  ids = group["id"]
  dates = group["date"]

  connections = group["connections"]
  correct_threads = []
  external_refs = {}  # Count external references
  self_refs = set()   # Track which nodes have self-reference

  # Build threads by tracing backwards and reversing
  # Connections show what each node replies to (backwards)
  # We need forward threads showing conversation flow
  # Use dict to group threads by starting node
  thread_groups = {}  # {start_node: [future_node1, future_node2, ...]}

  for idx, c in enumerate(connections):
    current_node_id = idx + 1  # Convert 0-based index to 1-based node ID

    # Filter: only keep backward references (nodes before current)
    backward_refs = [ref for ref in c if ref < current_node_id]

    if backward_refs:
      # Create forward thread by reversing: referenced_node -> current_node
      for ref_node in backward_refs:
        # Group by the starting node (ref_node)
        if ref_node not in thread_groups:
          thread_groups[ref_node] = []
        thread_groups[ref_node].append(current_node_id)

        # Count external reference
        if ref_node not in external_refs:
          external_refs[ref_node] = 1
        else:
          external_refs[ref_node] += 1
    else:
      # No backward refs means self-reference (starts new thread)
      self_refs.add(current_node_id)
      # DON'T add to thread_groups - only track in self_refs

  # Build complete conversation threads (root to leaf paths)
  # Create forward graph: node -> list of future nodes
  forward_graph = {}
  for start_node, future_nodes in thread_groups.items():
    forward_graph[start_node] = future_nodes

  # DEBUG: Print forward graph to verify connections
  print("\n=== FORWARD GRAPH DEBUG ===")
  for node_id in sorted(forward_graph.keys()):
    futures = forward_graph[node_id]
    print(f"Node {node_id} → Future nodes: {sorted(futures)}")
  print("===========================\n")

  # Find root nodes (nodes that are never referenced by other nodes)
  root_nodes = []
  all_referenced_nodes = set()
  for start_node, future_nodes in thread_groups.items():
    all_referenced_nodes.update(future_nodes)

  for node_id in ids:
    # A node is a root if it's not referenced by any other node
    # (Self-references don't count as being "referenced by others")
    if node_id not in all_referenced_nodes:
      root_nodes.append(node_id)

  # DEBUG: Print root nodes
  print("=== ROOT NODES DEBUG ===")
  print(f"Root nodes (not referenced by others): {sorted(root_nodes)}")
  print(f"All referenced nodes: {sorted(all_referenced_nodes)}")
  print("========================\n")

  # Build non-overlapping linear conversation threads
  # When a node has multiple future nodes (branching), we:
  # 1. End the current thread at that node
  # 2. Start new threads from each branch

  visited_nodes = set()

  def build_linear_thread(start_node, graph, self_refs_set):
    """Build a single linear thread starting from start_node, stopping at branches"""
    if start_node in visited_nodes:
      return [], []

    thread = [start_node]
    visited_nodes.add(start_node)
    current = start_node
    new_branch_starts = []

    while True:
      future_nodes = graph.get(current, [])
      # Filter out already visited nodes
      unvisited_futures = [n for n in future_nodes if n not in visited_nodes]

      if len(unvisited_futures) == 0:
        # No more future nodes - end thread
        break
      elif len(unvisited_futures) == 1:
        # Single path - continue thread
        next_node = unvisited_futures[0]
        thread.append(next_node)
        visited_nodes.add(next_node)
        current = next_node
      else:
        # Branching point - continue with first branch, queue the rest
        next_node = unvisited_futures[0]
        thread.append(next_node)
        visited_nodes.add(next_node)
        current = next_node

        # Queue remaining branches as new thread starts
        for branch in unvisited_futures[1:]:
          new_branch_starts.append(branch)

    # Format thread
    # Only use (node, node) format if truly isolated (no future nodes in graph)
    if len(thread) == 1 and start_node in self_refs_set and start_node not in graph:
      return [(start_node, start_node)], new_branch_starts
    else:
      return [tuple(thread)], new_branch_starts

  # Process all root nodes and branches
  queue = list(root_nodes)

  while queue:
    start = queue.pop(0)
    threads, new_branches = build_linear_thread(start, forward_graph, self_refs)
    correct_threads.extend(threads)
    queue.extend(new_branches)

  # Sort threads by starting node (first element of tuple)
  correct_threads.sort(key=lambda x: x[0])

  # Build node_counter based on the rules:
  # - If external refs > 0: counter = external refs
  # - If external refs == 0 and self-ref exists: counter = 1
  # - Otherwise: counter = 0
  node_counter = {}
  for node_id in ids:
    ext_count = external_refs.get(node_id, 0)
    has_self_ref = node_id in self_refs

    if ext_count > 0:
      node_counter[node_id] = ext_count
    elif has_self_ref:
      node_counter[node_id] = 1
    else:
      node_counter[node_id] = 0

  # construct a tree that can be used later to add extra input features(features will be associated with the node, ignore for now)
  # for now, the tree will just be used to model the message as a tree
  # for each node in succession, it will be full connected to the future nodes, for example
  # messages : [msg1, msg2, msg3, msg4] all sorted by time
  # msg1 is connected to msg2, msg3, msg4
  # msg2 is connected to msg3, msg4
  # msg4 is a leaf node

  # for each node in the graph, look at the connections, and in cases where a message has multiple links
  # based on how many links, this node will have a counter associated with it, where, when returned with a partial thread
  # decrement this counter, if it reaches zero, remove the node from the tree, otherwise, just decrement it

  # Build the message tree
  message_tree = _construct_fully_connected_tree(raw_messages, ids, dates, node_counter)

  return message_tree, correct_threads


def _construct_fully_connected_tree(raw_messages, ids, dates, node_counter):
  """
  Helper function to construct a fully connected message tree.

  Each message node is connected to all messages that come after it in time.
  For example, with messages [msg1, msg2, msg3, msg4]:
    - msg1 connects to msg2, msg3, msg4
    - msg2 connects to msg3, msg4
    - msg3 connects to msg4
    - msg4 is a leaf node

  Args:
    raw_messages: List of raw message strings
    ids: List of message IDs
    dates: List of message dates
    node_counter: Dictionary mapping node IDs to their reference counts

  Returns:
    List of message_node objects representing the tree
  """
  nodes = []

  # Step 1: Create all nodes
  for i, (msg_id, raw_msg, date) in enumerate(zip(ids, raw_messages, dates)):
    # Get the counter for this node (how many times it's referenced)
    # If not in node_counter, it means no one references it, so counter = 0
    counter = node_counter.get(msg_id, 0)
    node = message_node(msg_id, raw_msg, date, counter)
    nodes.append(node)

  # Step 2: Create fully connected forward edges
  # Each node at index i connects to all nodes at indices i+1, i+2, ..., n
  for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
      nodes[i].add_edge(nodes[j])

  # Print connections for all nodes
  # print("\n=== Message Tree Connections ===")
  for node in nodes:
    connected_ids = [edge.id for edge in node.forward_edges]
    # print(f"Node {node.id} (counter: {node.counter}) -> Connected to: {connected_ids}")
  # print("================================\n")

  return nodes


class message_node():
  # holds the main shape for each node in the graph, hence what parameters can be used to compute a score
  # for now, it will have the counter
  def __init__(self, node_id, raw_message, date, counter=1):
    self.id = node_id
    self.raw_message = raw_message
    self.date = date
    self.counter = counter  # how many times this node is referenced in the conversation threads
    self.forward_edges = []  # edges to all future nodes in the graph

  def add_edge(self, target_node):
    """Add an edge from this node to a future node"""
    self.forward_edges.append(target_node)

  def decrement_counter(self):
    """Decrement the counter when a partial thread is returned"""
    self.counter -= 1
    return self.counter 

def remove_used_nodes(message_tree, used_node_ids):
  """
  Remove or decrement nodes based on their usage in a partial thread.

  Args:
    message_tree: List of message_node objects
    used_node_ids: List of node IDs that were used in a partial thread (e.g., [1, 2, 4])

  Returns:
    tuple: (updated_message_tree, removed_nodes)
      - updated_message_tree: List of message_node objects with nodes removed or counters decremented
      - removed_nodes: List of message_node objects that were removed (counter reached 0)
  """
  nodes_to_remove = []

  # Process each used node
  for node_id in used_node_ids:
    # Find the node in the tree
    for node in message_tree:
      if node.id == node_id:
        # Decrement the counter
        remaining_count = node.decrement_counter()

        # If counter reaches 0, mark for removal
        if remaining_count <= 0:
          nodes_to_remove.append(node)
        break

  # Remove nodes that reached counter == 0
  for node in nodes_to_remove:
    message_tree.remove(node)
    # Also remove this node from other nodes' forward_edges
    for remaining_node in message_tree:
      if node in remaining_node.forward_edges:
        remaining_node.forward_edges.remove(node)

  return message_tree, nodes_to_remove

# displays the number of references a single node has relative to other nodes
def display_node_counter(message_tree, correct_threads):
  """
  Display the reference counter for each node in the message tree.

  Args:
    message_tree: List of message_node objects
    correct_threads: List of thread tuples showing conversation flow
  """
  if not display_counter:
    return

  print("\n" + "="*80)
  print("NODE COUNTER SUMMARY")
  print("="*80)
  print("Counter shows how many times each node is referenced by future nodes")
  print()

  for node in message_tree:
    # Get future nodes that reference this node from correct_threads
    future_refs = []
    for thread in correct_threads:
      if thread[0] == node.id and len(thread) > 1:
        # Thread starts with this node and has future nodes
        future_refs = list(thread[1:])
        break

    # Display node info
    if node.counter == 0:
      status = "Not referenced (leaf node)"
    elif node.counter == 1 and node.id in [t[0] for t in correct_threads if len(t) == 2 and t[0] == t[1]]:
      status = "Self-reference (starts new thread)"
    else:
      status = f"Referenced by {node.counter} future node(s): {future_refs}"

    print(f"Node {node.id} (counter: {node.counter}) - {status}")
    print(f"  Message: {node.raw_message[:80]}{'...' if len(node.raw_message) > 80 else ''}")

  print("="*80 + "\n")

# build_message_tree(example)