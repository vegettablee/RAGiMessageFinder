# Debug display configuration and functions for training loop

# ============================================================================
# DISPLAY FLAGS - Set to True/False to control what gets printed
# ============================================================================
SHOW_FULL_CONVERSATION = False      # Print all messages in the example
SHOW_RAW_CONNECTIONS = False        # Print raw connections from dataset
SHOW_GROUND_TRUTH_THREADS = True   # Print ground truth threads
SHOW_NODE_COUNTERS = False          # Print node counter summary
SHOW_EMBEDDING_INFO = False         # Print conversation embedding shapes
SHOW_THREAD_PREDICTIONS = True     # Print model predictions per thread

# Control how many examples/threads to show
DEBUG_FIRST_N_EXAMPLES = 2       # Only show debug for first N examples
DEBUG_FIRST_N_THREADS = 5          # Only show thread predictions for first N threads
MAX_THREADS_TO_DISPLAY = 15        # Max threads to show in ground truth


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_full_conversation(example_num, original_messages, ids):
    """Display all messages in the conversation"""
    if not SHOW_FULL_CONVERSATION:
        return

    print(f"\n{'='*80}")
    print(f"EXAMPLE {example_num} - FULL CONVERSATION ({len(original_messages)} messages)")
    print(f"{'='*80}")
    for i, msg in enumerate(original_messages):
        print(f"  Node {ids[i]}: {msg}")


def display_raw_connections(connections, ids):
    """Display raw connections from the dataset"""
    if not SHOW_RAW_CONNECTIONS:
        return

    print(f"\n{'='*80}")
    print(f"RAW CONNECTIONS (from dataset)")
    print(f"{'='*80}")
    for i, conn in enumerate(connections):
        print(f"  Node {ids[i]} connects to: {conn}")


def display_ground_truth_threads(correct_threads, original_messages):
    """Display ground truth threads with their messages"""
    if not SHOW_GROUND_TRUTH_THREADS:
        return

    print(f"\n{'='*80}")
    print(f"GROUND TRUTH THREADS ({len(correct_threads)} threads)")
    print(f"{'='*80}")
    print("Note: Each thread is (start_node, future_node1, future_node2, ...)")

    for thread_idx, thread in enumerate(correct_threads[:MAX_THREADS_TO_DISPLAY]):
        print(f"\nThread {thread_idx + 1}: {thread}")

        # Show what this thread represents
        start_node = thread[0]
        future_nodes = thread[1:] if len(thread) > 1 else []

        if start_node <= len(original_messages):
            print(f"  Start Node {start_node}: {original_messages[start_node - 1]}")

            if len(future_nodes) == 0 or (len(thread) == 2 and thread[0] == thread[1]):
                print(f"  → (Self-reference: starts new thread)")
            else:
                print(f"  Referenced by {len(future_nodes)} future node(s):")
                for fn in future_nodes:
                    if fn <= len(original_messages):
                        print(f"    - Node {fn}: {original_messages[fn - 1]}")

    if len(correct_threads) > MAX_THREADS_TO_DISPLAY:
        print(f"\n  ... and {len(correct_threads) - MAX_THREADS_TO_DISPLAY} more threads")
    print(f"{'='*80}\n")


def display_node_counters(message_tree, correct_threads):
    """Display node counter summary from construct_tree"""
    # Import the function from construct_tree
    from construct_tree import display_node_counter

    if SHOW_NODE_COUNTERS:
        display_node_counter(message_tree, correct_threads)


def display_embedding_info(messages, messages_emb, input_tensor):
    """Display conversation embedding information"""
    if not SHOW_EMBEDDING_INFO:
        return

    print(f"\n{'='*80}")
    print(f"CONVERSATION EMBEDDING INFO")
    print(f"{'='*80}")
    print(f"Conversation Length: {len(messages)} messages")
    print(f"Embedding Shape: {messages_emb.shape}")
    print(f"Input Tensor Shape: {input_tensor.shape}")
    print(f"{'='*80}\n")


def display_thread_prediction(thread_idx, correct_thread, pred_ids,
                               remaining_nodes, original_messages):
    """Display model predictions and decisions for a thread"""
    if not SHOW_THREAD_PREDICTIONS:
        return

    if thread_idx >= DEBUG_FIRST_N_THREADS:
        return

    print(f"\n{'='*80}")
    print(f"THREAD {thread_idx + 1} PREDICTION DEBUG")
    print(f"{'='*80}")
    print(f"Ground Truth Thread: {correct_thread}")

    print(f"\nModel Predictions (Top K):")
    print(f"  Predicted IDs: {pred_ids}")
    for i, pred_id in enumerate(pred_ids):
        msg = original_messages[pred_id - 1]
        truncated_msg = msg[:80] + '...' if len(msg) > 80 else msg
        print(f"    - Node {pred_id}: {truncated_msg}")

    print(f"\nRemaining State After This Thread:")
    print(f"  Remaining IDs: {remaining_nodes}")
    print(f"  Remaining Messages ({len(remaining_nodes)} total):")
    for rid in remaining_nodes[:5]:  # Show first 5
        msg = original_messages[rid - 1]
        truncated_msg = msg[:80] + '...' if len(msg) > 80 else msg
        print(f"    - Node {rid}: {truncated_msg}")
    if len(remaining_nodes) > 5:
        print(f"    ... and {len(remaining_nodes) - 5} more")
    print(f"{'='*80}\n")


def should_show_debug_for_example(batch_idx):
    """Check if we should show debug output for this example"""
    return batch_idx < DEBUG_FIRST_N_EXAMPLES
