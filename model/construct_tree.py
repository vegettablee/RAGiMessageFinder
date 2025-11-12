def build_message_tree(group):

  raw_messages = group["raw"]
  ids = group["id"]
  dates = group["date"]

  connections = group["connections"]
  correct_threads = []
  external_refs = {}  # Count external references
  self_refs = set()   # Track which nodes have self-references

  # Count references for each node
  for idx, c in enumerate(connections):
    current_node_id = idx + 1  # Convert 0-based index to 1-based node ID
    c_tuple = (current_node_id, c)
    correct_threads.append(c_tuple)

    # For each node referenced in the connection list
    for referenced_node in c:
      if referenced_node == current_node_id:
        # This is a self-reference
        self_refs.add(referenced_node)
      else:
        # This is an external reference
        if referenced_node not in external_refs:
          external_refs[referenced_node] = 1
        else:
          external_refs[referenced_node] += 1

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
  print("\n=== Message Tree Connections ===")
  for node in nodes:
    connected_ids = [edge.id for edge in node.forward_edges]
    print(f"Node {node.id} (counter: {node.counter}) -> Connected to: {connected_ids}")
  print("================================\n")

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

# build_message_tree(example)