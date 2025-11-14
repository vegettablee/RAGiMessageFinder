import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.optim as optim

from dataset_utils import load_dataset, get_dataloader, get_example
from construct_tree import build_message_tree, remove_used_nodes
from CD_model import CDModel
from loss_function import compute_loss_f1, compute_loss_BCE
import debug_display as dbg

EPOCHS = 1
BATCH_SIZE = 5 # since each thread technically gets 4 different model predictions
F1_THRESHOLD = 0.5
USE_CHECKPOINT = False
MAX_MESSAGES_PER_EXAMPLE = 20  # Limit conversation length for faster training
NUM_EXAMPLES = 1000
TEACHER_FORCE = True # when false, let the model choose freely

model = SentenceTransformer("all-mpnet-base-v2")
loss_fn = nn.BCEWithLogitsLoss() # binary cross entropy 

# all nodes are a direct reference to a real message
def training_loop():
  # Initialize models
  if USE_CHECKPOINT is False: 
    cd_model = CDModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # adam optimizer 
  
  dataset = load_dataset(split='train', num_examples=NUM_EXAMPLES)

  for epoch in range(EPOCHS):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print(f"{'='*80}\n")

    # Track metrics per epoch
    epoch_f1_scores = []
    epoch_accuracies = []
    epoch_losses = []

    dataloader = get_dataloader(dataset, batch_size=1, shuffle=True)
    for batch_idx, batch in enumerate(dataloader):
      for ex_idx, example in enumerate(batch):
      # get training example(might add pre-processing)
        example = batch[0]  # Extract example from batch

        # Truncate to first N messages for faster training
        if len(example["raw"]) > MAX_MESSAGES_PER_EXAMPLE:
          example = {
            "raw": example["raw"][:MAX_MESSAGES_PER_EXAMPLE],
            "id": example["id"][:MAX_MESSAGES_PER_EXAMPLE],
            "date": example["date"][:MAX_MESSAGES_PER_EXAMPLE],
            "connections": example["connections"][:MAX_MESSAGES_PER_EXAMPLE]
          }

        original_messages = example["raw"].copy() # keep a copy
        ids = example["id"] # [1,2,3,etc] direct index for nodes
        dates = example["date"] # contains the full date of each message
        connections = example["connections"] # original data used for ground truth, may not be needed
        message_tree, correct_threads = build_message_tree(example) # construct message tree with example

        # Debug display for first N examples
        if dbg.should_show_debug_for_example(batch_idx):
          dbg.display_full_conversation(batch_idx + 1, original_messages, ids)
          dbg.display_raw_connections(connections, ids)
          dbg.display_ground_truth_threads(correct_threads, original_messages)
          dbg.display_node_counters(message_tree, correct_threads)

        used_nodes = []
        msg_rmv = [] # messages to remove at each iteration
        remaining_nodes = ids.copy()
        messages = original_messages # use a copy

        # Accumulate loss and metrics across all threads in this example
        total_loss = 0
        example_f1_scores = []
        example_accuracies = []
        optimizer.zero_grad()

        for thread_idx, correct_thread in enumerate(correct_threads):
          if thread_idx == len(original_messages) - 1: # to handle cases like this where node 12 (counter: 0) -> connected to: []
            break
          # emb the entire conversation into one embedding as an input feature for the model
          # correct_thread is a tuple like (1, 1) or (2, 1), convert to 0-indexed
          correct_message_thread = [original_messages[i - 1] for i in correct_thread if i <= len(original_messages)]
          messages = example["raw"]
          messages_emb = model.encode(messages)

          num_possible_outputs = len(messages)

          # batch_size to 1 for initial testing
          input_tensor = torch.tensor(messages_emb).unsqueeze(0)

          # Debug: Show embedding info for first thread
          if dbg.should_show_debug_for_example(batch_idx) and thread_idx == 0:
            dbg.display_embedding_info(messages, messages_emb, input_tensor)

          thread_logits, keep_logits = cd_model(input_tensor) # shape [B, N]
          node_logits = thread_logits[0]
          thread_probs = torch.sigmoid(node_logits)
          top_k = len(correct_thread) # based on the length of the correct thread, return top k
          top_values, top_indices = torch.topk(thread_probs, k=top_k) 
          # change this later to only return the top scoring logits, i'm forcing top 4 initially for training, but once accuracy gets better
          # switch this 

          keep_probs = torch.sigmoid(keep_logits[0])

          selected_probs = keep_probs[top_indices]
          selected_ids = torch.tensor([ids[idx.item()] for idx in top_indices])
          mask = selected_probs > 0.75

          kept_probs = selected_probs[mask]
          kept_ids   = selected_ids[mask]

          # only select the kept_logits for what the model predicted, ignore all of the other values

          predicted = []
          pred_ids = []
          for idx in range(len(top_indices)):
            index = top_indices[idx].item()  # convert tensor to int
            predicted.append(messages[index])
            pred_ids.append(ids[index]) # ids that were predicted

          # Teacher forcing: remove ground truth nodes vs. model predictions
          if TEACHER_FORCE:
            # Remove nodes from ground truth (correct_thread) to guide training
            for node_id in correct_thread:
              if node_id in remaining_nodes:
                remaining_nodes.remove(node_id)
          else:
            # Remove nodes that model predicted (let model choose freely)
            for node_id in pred_ids:
              if node_id in remaining_nodes:
                remaining_nodes.remove(node_id)

          # Debug: Show thread prediction details
          if dbg.should_show_debug_for_example(batch_idx):
            dbg.display_thread_prediction(thread_idx, correct_thread, pred_ids, kept_probs,
                                          kept_ids, remaining_nodes, original_messages)
          # use a sigmoid activation to map the nodes to scores, and then match their indexes to properly do the loss function
          # form_prediction(logits, )  # TODO: incomplete
          # Use pred_ids (node IDs) instead of predicted (message strings) for F1 calculation
          f1_score = compute_loss_f1(pred_ids, remaining_nodes, list(correct_thread))
          used_nodes.append(correct_thread)

          message_tree, removed_nodes = remove_used_nodes(message_tree, used_nodes)
          msg_rmv = used_nodes

          for idx, n in enumerate(removed_nodes):
            rm_idx = ids.index(n) # find index based on node number
            del messages[rm_idx] # remove from messages
            msg_rmv = [] # reset

          correct_nodes = set(correct_thread)
          pred_ids = set(pred_ids) # turn into set
          true_labels = []
          for idx in range(len(node_logits)): # create truth labels for BCE loss function
          # Map logit position to node ID
            current_id = ids[idx]
            if current_id in correct_nodes:
              true_labels.append(1)
            else:
              true_labels.append(0)

          loss = loss_fn(node_logits, torch.tensor(true_labels, dtype=torch.float32))
          total_loss += loss

          # Calculate accuracy: intersection of predicted and ground truth
          pred_set = set(pred_ids)
          correct_set = set(correct_thread)
          if len(pred_set) > 0:
            accuracy = len(pred_set.intersection(correct_set)) / len(pred_set)
          else:
            accuracy = 0.0

          # Track metrics
          example_f1_scores.append(f1_score)
          example_accuracies.append(accuracy)

        # Backpropagate after all threads in this example
        if total_loss > 0:
          total_loss.backward()
          optimizer.step()

          # example-level metrics
          avg_f1 = sum(example_f1_scores) / len(example_f1_scores) if example_f1_scores else 0.0
          avg_accuracy = sum(example_accuracies) / len(example_accuracies) if example_accuracies else 0.0

          # epoch metrics
          epoch_f1_scores.extend(example_f1_scores)
          epoch_accuracies.extend(example_accuracies)
          epoch_losses.append(total_loss.item())

          # Print example summary
          print(f"Example {batch_idx + 1}/{len(dataset)} | Loss: {total_loss.item():.4f} | Avg F1: {avg_f1:.4f} | Avg Accuracy: {avg_accuracy:.4f}")

    # Print epoch summary
    epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    epoch_avg_f1 = sum(epoch_f1_scores) / len(epoch_f1_scores) if epoch_f1_scores else 0.0
    epoch_avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0.0
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1} SUMMARY")
    print(f"{'='*80}")
    print(f"Average Loss:     {epoch_avg_loss:.4f}")
    print(f"Average F1 Score: {epoch_avg_f1:.4f}")
    print(f"Average Accuracy: {epoch_avg_accuracy:.4f}")
    print(f"{'='*80}\n")

    # Save epoch data
    save_data(cd_model, optimizer, epoch + 1, epoch_avg_loss, epoch_avg_f1, epoch_avg_accuracy)


def save_data(cd_model, optimizer, epoch_num, avg_loss, avg_f1, avg_accuracy):
  """
  Save training run data including model config and metrics.

  Args:
    cd_model: The CDModel instance
    optimizer: The optimizer being used
    epoch_num: Current epoch number
    avg_loss: Average loss for the epoch
    avg_f1: Average F1 score for the epoch
    avg_accuracy: Average accuracy for the epoch
  """
  import json
  from datetime import datetime

  # Get model architecture config
  model_config = cd_model.get_model_config()

  # Get learning rate from optimizer
  learning_rate = optimizer.param_groups[0]['lr']

  # Compile all run information
  run_data = {
    'timestamp': datetime.now().isoformat(),
    'epoch': epoch_num,
    'metrics': {
      'avg_loss': avg_loss,
      'avg_f1_score': avg_f1,
      'avg_accuracy': avg_accuracy
    },
    'model_config': model_config,
    'training_config': {
      'learning_rate': learning_rate,
      'optimizer': type(optimizer).__name__,
      'loss_function': 'BCEWithLogitsLoss',
      'max_messages_per_example': MAX_MESSAGES_PER_EXAMPLE,
      'num_examples': NUM_EXAMPLES,
      'batch_size': BATCH_SIZE,
      'f1_threshold': F1_THRESHOLD
    }
  }

  # Save to file
  filename = f"results.json"
  with open(filename, 'w') as f:
    json.dump(run_data, f, indent=2)

  print(f"Saved training data to {filename}")
  return run_data 

training_loop() 