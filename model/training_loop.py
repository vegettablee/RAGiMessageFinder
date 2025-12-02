import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder.losses import ListNetLoss
import torch.optim as optim

from dataset_utils import load_dataset, get_dataloader, get_example
from construct_tree import build_message_tree, remove_used_nodes
from CD_model import CDModel
from loss_function import compute_loss_f1, ListNetLoss
from save import save_model, save_checkpoint
import debug_display as dbg
import numpy as np
import random

EPOCHS = 30 # each epoch is roughly 153 examples
BATCH_SIZE = 5 # since each thread technically gets 4 different model predictions
F1_THRESHOLD = 0.5
USE_CHECKPOINT = False
SAVE_LAST_CHECKPOINT = False # saves a checkpoint after all epochs are finished 
SAVE_MODEL = False
MAX_MESSAGES_PER_EXAMPLE = 17 # limit conversation length for faster training
NUM_EXAMPLES = 5000 # number of examples to load, but not necessarily use 
TEACHER_FORCE = True # when false, let the model choose freely
PRUNE_SELF_NODES_PROB = 0.75 # only keep 25 percent of nodes who do not have any connections 
PRUNE_LONG_NODES_PROB = 1.1 # cut all examples with thread length > 8 
PRUNE_LONG_NODE_LEN = 8 # minimum node length to prune 
INCLUDE_LAST_EXAMPLE = False # if last example is a lone-node, skip it to see pure accuracy 
SIGMOID_THRESHOLD = 0.50 # 0.75 because I want the model to be confident about it's choices
TEMPERATURE = 0.5 # used to make the keep_head have a sharper decision boundary 
RANK_HEAD_WEIGHT = 1.0 # weighted scale factor for loss for rank head 
KEEP_HEAD_WEIGHT = 1.25 # weighted scale factor for loss for keep head 
# keep_head_weight trials : 1.5, 2.0, 1.25
# temperate trials : 0.75, 0.5, 0.5
model = SentenceTransformer("all-mpnet-base-v2")

loss_fn = ListNetLoss() # ListNetLoss for first output head for ranking nodes 
keep_loss_fn = nn.BCEWithLogitsLoss() # BCEloss for second output head for keeping/discarding nodes

# all nodes are a direct reference to a real message
def training_loop():
  # Initialize models
  if USE_CHECKPOINT is False: 
    cd_model = CDModel()
    optimizer = optim.Adam(cd_model.parameters(), lr=0.001) # adam optimizer
  
  train_dataset = load_dataset(split='train', num_examples=NUM_EXAMPLES)
  dev_dataset = load_dataset(split='dev', num_examples=NUM_EXAMPLES)
  # use these metrics keep track of class imbalances with the dataset
  total_threads = 0 
  num_long_threads = 0
  num_lone_threads = 0 
  num_long_pruned = 0 
  num_lone_pruned = 0
  total_examples = 0 

  for epoch in range(EPOCHS):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print(f"{'='*80}\n")

    # track metrics per epoch
    epoch_f1_scores = []
    epoch_accuracies = []
    epoch_losses = []
    epoch_precision_scores = []
    epoch_recall_scores = []
    epoch_specificity_scores = []

    # Rank head epoch metrics
    epoch_rank_f1_scores = []
    epoch_rank_accuracies = []
    epoch_rank_precision_scores = []
    epoch_rank_recall_scores = []
    epoch_rank_specificity_scores = []

    # Keep head epoch metrics
    epoch_keep_f1_scores = []
    epoch_keep_accuracies = []
    epoch_keep_precision_scores = []
    epoch_keep_recall_scores = []
    epoch_keep_specificity_scores = []

    dataloader = get_dataloader(train_dataset, batch_size=1, shuffle=True)
    for batch_idx, batch in enumerate(dataloader):
      for ex_idx, example in enumerate(batch):
        # get training example(might add pre-processing)
        total_examples += 1
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
        messages = original_messages.copy() # use a copy, don't ever mutate

        #print("Messages before removing : " + str(messages))
        #print("Threads before removing : " + str(correct_threads) + "\n\n")
        total_threads += len(correct_threads)
        for thread_idx, thread in enumerate(correct_threads):
          if len(thread) == 1: 
            rand = random.uniform(0.0, 1.0)
            num_lone_threads += 1
            if(rand < PRUNE_SELF_NODES_PROB): 
              message_tree, removed = remove_used_nodes(message_tree, thread)
              root_id = removed[0].id
              # print("root id : " + str(root_id))
              ids.remove(root_id)
              correct_threads.remove(thread)
              num_lone_pruned += 1
          if len(thread) > PRUNE_LONG_NODE_LEN: # sometimes prune nodes with exceedingly greater length 
            num_long_threads += 1 
            rand = random.uniform(0.0, 1.0)
            if(rand < PRUNE_LONG_NODES_PROB): 
              message_tree, removed = remove_used_nodes(message_tree, thread)
              for node in removed: 
                ids.remove(node.id)
              correct_threads.remove(thread)
              num_long_pruned += 1
              # remove redundant threads 
        #print("Messages after removing : " + str(messages))
        #print("Threads after removing : " + str(correct_threads))
        
        # Debug display for first N examples
        if dbg.should_show_debug_for_example(batch_idx):
          dbg.display_full_conversation(batch_idx + 1, original_messages, ids)
          dbg.display_raw_connections(connections, ids)
          dbg.display_ground_truth_threads(correct_threads, original_messages)
          dbg.display_node_counters(message_tree, correct_threads)
        
        used_nodes = []
        msg_rmv = [] # messages to remove at each iteration
        remaining_nodes = ids.copy()

        # Accumulate loss and metrics across all threads in this example
        total_loss = 0
        example_f1_scores = []
        example_accuracies = []
        example_precision_scores = []
        example_recall_scores = []
        example_specificity_scores = []

        # Rank head metrics
        example_rank_f1_scores = []
        example_rank_accuracies = []
        example_rank_precision_scores = []
        example_rank_recall_scores = []
        example_rank_specificity_scores = []

        # Keep head metrics
        example_keep_f1_scores = []
        example_keep_accuracies = []
        example_keep_precision_scores = []
        example_keep_recall_scores = []
        example_keep_specificity_scores = []

        optimizer.zero_grad()

        num_threads_skipped = 0 

        for thread_idx, correct_thread in enumerate(correct_threads):
          if thread_idx == len(original_messages) - 1: # to handle cases like this where node 12 (counter: 0) -> connected to: []
            break
          if thread_idx == len(correct_threads) - 1: # on last thread
            if len(correct_thread) == 1: # only keep if thread is not a lone-node
              num_threads_skipped += 1
              #print("skipped last lone thread : " + str(correct_thread))
              break
          # emb the entire conversation into one embedding as an input feature for the model
          # correct_thread is a tuple like (1, 1) or (2, 1), convert to 0-indexed
          current_messages = [] 
          for node_id in remaining_nodes: 
            current_messages.append(messages[node_id - 1])

          messages_emb = model.encode(current_messages)

          num_possible_outputs = len(current_messages)
          # print("total of number possible messages for model to pick " + str(num_possible_outputs))
          # batch_size to 1 for initial testing
          input_tensor = torch.tensor(messages_emb).unsqueeze(0)

          # Debug: Show embedding info for first thread
          if dbg.should_show_debug_for_example(batch_idx) and thread_idx == 0:
            dbg.display_embedding_info(current_messages, messages_emb, input_tensor)

          thread_logits, keep_logits = cd_model(input_tensor) # shape [B, N]
          # print("Shape of keep logits " + str(keep_logits.shape))

          keep_node_logits = keep_logits[0]

          keep_node_probs = torch.sigmoid(keep_node_logits / TEMPERATURE)
          top_keep_indices = [] 
          pred_keep_ids = []
          pred_kept_messages = [] 
          has_kept_pred = False
          for idx, i in enumerate(keep_node_probs): # get all messages above sigmoid threshold 
            if i >= SIGMOID_THRESHOLD: 
              top_keep_indices.append(idx)
              has_kept_pred = True
          
          if has_kept_pred is True: # there was a sigmoid value above threshold
            for idx in top_keep_indices:
              pred_kept_id = remaining_nodes[idx]
              pred_keep_ids.append(pred_kept_id)
              pred_kept_messages.append(messages[pred_kept_id - 1])

          # print("Kept ids " + str(pred_keep_ids))
              
          node_logits = thread_logits[0]
          # print(str(node_logits))
          thread_probs = torch.softmax(node_logits, dim=-1) # use softmax because of ListNet loss instead of BCE
          top_k = min(len(correct_thread), len(thread_probs)) # ensure k doesn't exceed available elements
          top_values, top_indices = torch.topk(thread_probs, k=top_k) 
         
          predicted = []
          pred_ids = []
          for idx in range(len(top_indices)):
            index = top_indices[idx].item()  # convert tensor to int
            pred_id = remaining_nodes[index] # find index of id predicted, the map to original message array
            predicted.append(messages[pred_id - 1])
            pred_ids.append(pred_id) # ids that were predicted

          rank_f1_score, rank_precision, rank_recall, rank_specificity = compute_loss_f1(pred_ids, remaining_nodes, list(correct_thread))
          keep_f1_score, keep_precision, keep_recall, keep_specificy = compute_loss_f1(pred_keep_ids, remaining_nodes, list(correct_thread))
          

          used_nodes = list(correct_thread)

          correct_nodes = set(correct_thread)
          set_pred_ids = set(pred_ids) # turn into set
          true_labels = [] # truth labels for ListNetLoss
          kept_truth_labels = [] # truth labels for second output head
          rank_range = len(correct_thread) # first used nodes get higher rank for model to learn importance 
          for i, node_id in enumerate(remaining_nodes): # create truth labels for listnet loss function
          # Map logit position to node ID
            if node_id in correct_nodes:
              true_labels.append(rank_range)
              kept_truth_labels.append(1)
              rank_range += -1 # decrement 
            else:
              true_labels.append(0)
              kept_truth_labels.append(0) 

          #print("Rank truth labels : " + str(true_labels))
          #print("Kept truth labels : " + str(kept_truth_labels))

          loss = loss_fn(node_logits, torch.tensor(true_labels, dtype=torch.float32))
          keep_loss = keep_loss_fn(keep_node_logits, torch.tensor(kept_truth_labels, dtype=torch.float32))

          # trying a weighted sum, keep_head is more inaccurate in comparison to the ranking head 
          total_loss += loss * RANK_HEAD_WEIGHT + keep_loss * KEEP_HEAD_WEIGHT # add both losses as one representation 

          # calculate accuracy: intersection of predicted and ground truth
          # accuracy is determined by whether the model predicted correct nodes as well as had the correct order 
          total_correct = 0
          for idx, pred in enumerate(pred_ids): 
            if pred_ids[idx] == correct_thread[idx]:
              total_correct += 1 
    
          if total_correct == 0: # no correct predictions
            accuracy = 0
          else: 
            accuracy = total_correct / len(correct_thread)
          
          kept_correct = 0 # calculate the accuracy for the kept nodes
          if has_kept_pred is True: 
            kept_correct = 0 
            for idx, pred in enumerate(pred_keep_ids): 
              if pred in correct_nodes: 
                kept_correct += 1
            kept_accuracy = kept_correct / len(correct_thread)
          else: 
            kept_accuracy = 0 

          general_accuracy = (accuracy + kept_accuracy) / 2
          general_f1 = (rank_f1_score + keep_f1_score) / 2
          general_precision = (rank_precision + keep_precision) / 2
          general_recall = (rank_recall + keep_recall) / 2
          general_specificity = (rank_specificity + keep_specificy) / 2 


          # track general metrics (combined from both heads)
          example_f1_scores.append(general_f1)
          example_accuracies.append(general_accuracy)
          example_precision_scores.append(general_precision)
          example_recall_scores.append(general_recall)
          example_specificity_scores.append(general_specificity)

          # track rank head metrics
          example_rank_f1_scores.append(rank_f1_score)
          example_rank_accuracies.append(accuracy)
          example_rank_precision_scores.append(rank_precision)
          example_rank_recall_scores.append(rank_recall)
          example_rank_specificity_scores.append(rank_specificity)

          # track keep head metrics
          example_keep_f1_scores.append(keep_f1_score)
          example_keep_accuracies.append(kept_accuracy)
          example_keep_precision_scores.append(keep_precision)
          example_keep_recall_scores.append(keep_recall)
          example_keep_specificity_scores.append(keep_specificy)
          
          # remove used nodes from ids 
          # remove used messages from messages for next iteration 
          message_tree, removed = remove_used_nodes(message_tree, used_nodes) # removed has shape [message_node(id=1, counter=0)]
          removed_nodes = [item.id for item in removed] # removed_nodes has shape [1,2,3]
          msg_rmv = used_nodes

          # Filter out removed nodes from both messages and ids for next iteration
          removed_set = set(removed_nodes)
          ids = [node_id for node_id in ids if node_id not in removed_set]
          # Teacher forcing: remove ground truth nodes vs. model predictions
          if TEACHER_FORCE is True:
            # Remove nodes from ground truth (correct_thread) to guide training
            for node_id in correct_thread:
              if node_id in remaining_nodes:
                remaining_nodes.remove(node_id)
          else:
            # remove nodes that model predicted (let model choose freely)
            for node_id in pred_ids:
              if node_id in remaining_nodes:
                remaining_nodes.remove(node_id)
          
          if dbg.should_show_debug_for_example(batch_idx):
            dbg.display_thread_prediction(thread_idx, correct_thread, pred_ids, remaining_nodes, original_messages)

        # backpropagate after all threads in this example
        if total_loss > 0:
          total_loss.backward()
          optimizer.step()

          # general metrics for both output heads
          num_valid_threads = len(correct_threads) - num_threads_skipped
          avg_f1 = sum(example_f1_scores) / num_valid_threads if num_valid_threads > 0 else 0.0
          avg_accuracy = sum(example_accuracies) / num_valid_threads if num_valid_threads > 0 else 0.0
          average_loss = total_loss.item() / num_valid_threads if num_valid_threads > 0 else 0.0

          # rank head metrics
          rank_avg_f1 = sum(example_rank_f1_scores) / num_valid_threads if num_valid_threads > 0 else 0.0
          rank_avg_accuracy = sum(example_rank_accuracies) / num_valid_threads if num_valid_threads > 0 else 0.0

          # keep head metrics
          keep_avg_f1 = sum(example_keep_f1_scores) / num_valid_threads if num_valid_threads > 0 else 0.0
          keep_avg_accuracy = sum(example_keep_accuracies) / num_valid_threads if num_valid_threads > 0 else 0.0

          # accumulate general metrics to epoch
          epoch_f1_scores.extend(example_f1_scores)
          epoch_accuracies.extend(example_accuracies)
          epoch_precision_scores.extend(example_precision_scores)
          epoch_recall_scores.extend(example_recall_scores)
          epoch_specificity_scores.extend(example_specificity_scores)
          epoch_losses.append(average_loss)

          # accumulate rank head metrics to epoch
          epoch_rank_f1_scores.extend(example_rank_f1_scores)
          epoch_rank_accuracies.extend(example_rank_accuracies)
          epoch_rank_precision_scores.extend(example_rank_precision_scores)
          epoch_rank_recall_scores.extend(example_rank_recall_scores)
          epoch_rank_specificity_scores.extend(example_rank_specificity_scores)

          # accumulate keep head metrics to epoch
          epoch_keep_f1_scores.extend(example_keep_f1_scores)
          epoch_keep_accuracies.extend(example_keep_accuracies)
          epoch_keep_precision_scores.extend(example_keep_precision_scores)
          epoch_keep_recall_scores.extend(example_keep_recall_scores)
          epoch_keep_specificity_scores.extend(example_keep_specificity_scores)

          # Print example summary
          print(f"Example general average {batch_idx + 1}/{len(train_dataset)} | Loss: {average_loss} | Avg F1: {avg_f1:.4f} | Avg Accuracy: {avg_accuracy:.4f}")

    # Print epoch summary - general metrics
    epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    epoch_avg_f1 = sum(epoch_f1_scores) / len(epoch_f1_scores) if epoch_f1_scores else 0.0
    epoch_avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0.0
    epoch_avg_precision = sum(epoch_precision_scores) / len(epoch_precision_scores) if epoch_precision_scores else 0.0
    epoch_avg_recall = sum(epoch_recall_scores) / len(epoch_recall_scores) if epoch_recall_scores else 0.0
    epoch_avg_specificity = sum(epoch_specificity_scores) / len(epoch_specificity_scores) if epoch_specificity_scores else 0.0

    # Rank head epoch averages
    epoch_rank_avg_f1 = sum(epoch_rank_f1_scores) / len(epoch_rank_f1_scores) if epoch_rank_f1_scores else 0.0
    epoch_rank_avg_accuracy = sum(epoch_rank_accuracies) / len(epoch_rank_accuracies) if epoch_rank_accuracies else 0.0
    epoch_rank_avg_precision = sum(epoch_rank_precision_scores) / len(epoch_rank_precision_scores) if epoch_rank_precision_scores else 0.0
    epoch_rank_avg_recall = sum(epoch_rank_recall_scores) / len(epoch_rank_recall_scores) if epoch_rank_recall_scores else 0.0
    epoch_rank_avg_specificity = sum(epoch_rank_specificity_scores) / len(epoch_rank_specificity_scores) if epoch_rank_specificity_scores else 0.0

    # Keep head epoch averages
    epoch_keep_avg_f1 = sum(epoch_keep_f1_scores) / len(epoch_keep_f1_scores) if epoch_keep_f1_scores else 0.0
    epoch_keep_avg_accuracy = sum(epoch_keep_accuracies) / len(epoch_keep_accuracies) if epoch_keep_accuracies else 0.0
    epoch_keep_avg_precision = sum(epoch_keep_precision_scores) / len(epoch_keep_precision_scores) if epoch_keep_precision_scores else 0.0
    epoch_keep_avg_recall = sum(epoch_keep_recall_scores) / len(epoch_keep_recall_scores) if epoch_keep_recall_scores else 0.0
    epoch_keep_avg_specificity = sum(epoch_keep_specificity_scores) / len(epoch_keep_specificity_scores) if epoch_keep_specificity_scores else 0.0

    thread_metrics = {
    "percentage_long_pruned" : (num_long_threads - num_long_pruned) / total_threads, # percentage kept relative to pruned for long threads
    "percentage_lone_pruned" : (num_lone_threads - num_lone_pruned) / total_threads, # percentage kept relative to prune for lone threads 
    "percentage_long_threads" : (num_long_threads / total_threads), # percentage of long threads relative to the entire dataset
    "percentage_lone_threads" : num_lone_threads / total_threads, # percentage of lone threads relative to the entire dataset 
    }
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1} SUMMARY")
    print(f"{'='*80}")
    print(f"Average Loss:        {epoch_avg_loss:.4f}")
    print(f"Average F1 Score:    {epoch_avg_f1:.4f}")
    print(f"Average Accuracy:    {epoch_avg_accuracy:.4f}")
    print(f"Average Precision:   {epoch_avg_precision:.4f}")
    print(f"Average Recall:      {epoch_avg_recall:.4f}")
    print(f"Average Specificity: {epoch_avg_specificity:.4f}")
    print(f"{'='*80}\n")

    # Save epoch data
    rank_head_metrics = {
      'avg_f1_score': epoch_rank_avg_f1,
      'avg_accuracy': epoch_rank_avg_accuracy,
      'avg_precision': epoch_rank_avg_precision,
      'avg_recall': epoch_rank_avg_recall,
      'avg_specificity': epoch_rank_avg_specificity,
    }
    keep_head_metrics = {
      'avg_f1_score': epoch_keep_avg_f1,
      'avg_accuracy': epoch_keep_avg_accuracy,
      'avg_precision': epoch_keep_avg_precision,
      'avg_recall': epoch_keep_avg_recall,
      'avg_specificity': epoch_keep_avg_specificity,
    }
    save_data(cd_model, optimizer, epoch + 1, epoch_avg_loss, epoch_avg_f1, epoch_avg_accuracy, epoch_avg_precision, epoch_avg_recall, epoch_avg_specificity, rank_head_metrics, keep_head_metrics, thread_metrics, total_examples)

  # Save checkpoint after all epochs complete
  if SAVE_LAST_CHECKPOINT:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filepath = f"checkpoints/checkpoint_{timestamp}_epoch{EPOCHS}_f1{epoch_avg_f1:.3f}.pt"
    save_checkpoint(cd_model, optimizer, EPOCHS, epoch_avg_loss, filepath=checkpoint_filepath)
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - Checkpoint saved to {checkpoint_filepath}")
    print(f"Final F1: {epoch_avg_f1:.4f} | Final Loss: {epoch_avg_loss:.4f}")
    print(f"{'='*80}\n")

def remove_messages(ids, messages): 
  return messages

def save_data(cd_model, optimizer, epoch_num, avg_loss, avg_f1, avg_accuracy, avg_precision, avg_recall, avg_specificity, rank_head_metrics, keep_head_metrics, thread_metrics, total_examples):

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
    'all_metrics': {
      'avg_loss': avg_loss,
      'avg_f1_score': avg_f1,
      'avg_accuracy': avg_accuracy,
      'avg_precision': avg_precision,
      'avg_recall': avg_recall,
      'avg_specificity': avg_specificity,
    },
    "rank_head_metrics" : {
      'avg_loss': avg_loss,  # Same loss for both heads
      'avg_f1_score': rank_head_metrics['avg_f1_score'],
      'avg_accuracy': rank_head_metrics['avg_accuracy'],
      'avg_precision': rank_head_metrics['avg_precision'],
      'avg_recall': rank_head_metrics['avg_recall'],
      'avg_specificity': rank_head_metrics['avg_specificity'],
    },
    "keep_head_metrics" : {
      'avg_loss': avg_loss,  # Same loss for both heads
      'avg_f1_score': keep_head_metrics['avg_f1_score'],
      'avg_accuracy': keep_head_metrics['avg_accuracy'],
      'avg_precision': keep_head_metrics['avg_precision'],
      'avg_recall': keep_head_metrics['avg_recall'],
      'avg_specificity': keep_head_metrics['avg_specificity'],
    }, 
    'thread_metrics' : { 
      'percentage_kept_long_threads' : thread_metrics["percentage_long_pruned"], 
      'percentage_kept_lone_threads' : thread_metrics["percentage_lone_pruned"],
      'percentage_total_long_threads' : thread_metrics["percentage_long_threads"], 
      'percentage_total_lone_threads' : thread_metrics["percentage_lone_threads"]
    },
    'model_config': model_config,
    'training_config': {
      'learning_rate': learning_rate,
      'optimizer': type(optimizer).__name__,
      'rank_loss_function': 'ListNetLoss',
      "keep_loss_function" : "BCELoss",
      'max_messages_per_example': MAX_MESSAGES_PER_EXAMPLE,
      'num_examples': total_examples,
      'batch_size': BATCH_SIZE,
      'f1_threshold': F1_THRESHOLD, 
      'using_attention' : "TRUE", 
    }
  }

  # Save to file
  filename = f"results.json"

  # Read existing results if file exists
  try:
    with open(filename, 'r') as f:
      results = json.load(f)
      if not isinstance(results, list):
        results = [results]  # Convert old format to list
  except (FileNotFoundError, json.JSONDecodeError):
    results = []

  # Append new run data
  results.append(run_data)

  # Write updated results back
  with open(filename, 'w') as f:
    json.dump(results, f, indent=2)

  print(f"Saved training data to {filename}")
  return run_data 

training_loop() 