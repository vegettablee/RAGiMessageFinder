import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.optim as optim

from dataset_utils import load_dataset, get_dataloader, get_example
from construct_tree import build_message_tree, remove_used_nodes
from CD_model import CDModel
from loss_function import compute_loss_f1, compute_loss_BCE
from dataset_utils import get_dataloader, load_dataset

EPOCHS = 10
BATCH_SIZE = 32
F1_THRESHOLD = 0.5 
USE_CHECKPOINT = False 
DATASET_NAME = ""
NUM_EXAMPLES = 10

model = SentenceTransformer("all-mpnet-base-v2")
loss_fn = nn.BCEWithLogitsLoss() # binary cross entropy 

# all nodes are a direct reference to a real message
def training_loop():
  # Initialize models
  if USE_CHECKPOINT is False: 
    cd_model = CDModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # adam optimizer 
  


  for epoch in range(EPOCHS): 
    if epoch == 2: 
      break
    for example_idx in range(NUM_EXAMPLES): # replace this to for batch_idx, batch in enumerate(dataloader) after initial testing
  # get training example(might add pre-processing)
      example = get_example()
      original_messages = example["raw"].copy() # keep a copy
      ids = example["id"] # [1,2,3,etc] direct index for nodes
      dates = example["date"] # contains the full date of each message
      connections = example["connections"] # original data used for ground truth, may not be needed

      # construct message tree with example
      message_tree, correct_threads = build_message_tree(example)

      used_nodes = []
      msg_rmv = [] # messages to remove at each iteration
      remaining_nodes = ids.copy()
      messages = original_messages # use a copy

      for idx, correct_thread in enumerate(correct_threads):
        if idx == len(original_messages) - 1: # to handle cases like this where node 12 (counter: 0) -> connected to: []
          break
        # emb the entire conversation into one embedding as an input feature for the model
        correct_message_thread = [original_messages[i] for i in correct_thread]
        messages = example["raw"]
        messages_emb = model.encode(messages)

        num_possible_outputs = len(messages)

        # batch_size to 1 for initial testing
        input_tensor = torch.tensor(messages_emb).unsqueeze(0)
        print("Input tensor shape : " + str(input_tensor.shape))

        logits = cd_model(input_tensor) # shape [B, N]
        node_logits = logits[0]
        probs = torch.sigmoid(node_logits)
        top_values, top_indices = torch.topk(probs, k=3)

        predicted = []
        pred_ids = []
        for idx in range(len(top_indices)):
          index = top_indices[idx].item()  # convert tensor to int
          predicted.append(messages[index])
          pred_ids.append(ids[index]) # ids that were predicted 
          # map index to node ID and remove from remaining nodes
          node_id = ids[index]
          if node_id in remaining_nodes:
            remaining_nodes.remove(node_id)

        # use a sigmoid activation to map the nodes to scores, and then match their indexes to properly do the loss function 
        # form_prediction(logits, )  # TODO: incomplete
        f1_score = compute_loss_f1(predicted, remaining_nodes, correct_thread)
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
          
        optimizer.zero_grad()
        loss = loss_fn(node_logits, torch.tensor(true_labels, dtype=torch.float32))
        loss.backward() 
        optimizer.step()

        print("Ground truth thread messages : " + str(correct_message_thread))
        print("Ground truth node ids : " + str(correct_thread))
        print("Messages emb shape : " + str(messages_emb.shape)) 
        print("Remaining node ids : " + str(remaining_nodes))
        print("Model predicted : " + str(predicted))
        print("F1 score : " + str(f1_score))
        print("True labels " + str(true_labels) + "\n\n")


training_loop() 