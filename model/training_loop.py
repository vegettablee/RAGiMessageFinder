import torch as nn
import torch

from dataset_utils import load_dataset, get_dataloader, get_example
from construct_tree import build_message_tree, remove_used_nodes
from CD_model import CDModel
from loss_function import compute_loss

EPOCHS = 10 
BATCH_SIZE = 32 
BCE_THRESHOLD = 0.5 

# all nodes are a direct reference to a real message 
def training_loop(): 
  # get training example(might add pre-processing) 
  example = get_example()
  original_messages = example["raw"] # keep a copy 
  ids = example["id"] # [1,2,3,etc] direct index for nodes
  dates = example["date"] # contains the full date of each message 
  connections = example["connections"] # original data used for ground truth, may not be needed

  # construct message tree with example 
  message_tree, correct_threads = build_message_tree(example) 

  used_nodes = []
  msg_rmv = [] # messages to remove at each iteration 
  remaining_nodes = ids 
  messages = original_messages # use a copy

  for idx, thread in enumerate(correct_threads): 
    # emb the entire conversation at one 
    messages = example["raw"] 
    messages_emb = model.encode(messages) 
    num_possible_outputs = len(messages)
    # batch_size to 1 for initial testing
    input_tensor = torch.tensor(1, messages_emb, num_possible_outputs)

    logits = CDModel(input_tensor) # shape [B, N]
    node_logits = logits[0]

    loss = compute_loss(node_logits, ids, BCE_THRESHOLD)

    used_nodes.append(correct_threads)
    message_tree, removed_nodes = remove_used_nodes(message_tree, used_nodes)
    msg_rmv = used_nodes

    for idx, n in enumerate(removed_nodes): 
      rm_idx = ids.index(n) # find index based on node number
      messages.remove(rm_idx) # remove from messages
      msg_rmv = [] # reset 


  # get the top-logits with score > 0.5, handle the case where there are no top-logits chosen








  


