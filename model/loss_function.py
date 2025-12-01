import torch
import torch.nn.functional as F

# f1 in this context is getting used to measure how accurate the model is choosing nodes, regardless of order
def compute_loss_f1(predicted, all_node_choices, correct_nodes): 
      
  all_node_set = set(all_node_choices)
  predicted_set = set(predicted)
  correct_set = set(correct_nodes)
    
  TP = len(predicted_set & correct_set)
  FP = len(predicted_set - correct_set)
  FN = len(correct_set - predicted_set)
  TN = len((all_node_set - predicted_set) & (all_node_set - correct_set))
  
  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  recall = TP / (TP + FN) if (TP + FN) > 0 else 0
  specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
  return f1, precision, recall, specificity
# listnetloss uses categorical distributions to compute losses that take into account ranking 
# categorical distributions refer to anything where a random variable can take on different, mutually exclusive categories
# can handle multiple outcomes, uses discrete values because in this case we are measuring which node is chosen
# softmax distribution is used for the model logits, this creates a ranking where all values must sum to 1 
# categorical distribution with normalization is used with the truth labels, so at each position, it should be compared with what 
# the model chose, for example 
# model has values : [0.21, 0.5, 0.29]
# truth has values : [0, 0.5, 0.5]
# the model should be ranking the last 2 values similarly because the normalized truth distribution says that the model should rank 
# this logit with percentage being chosen as 50 percent 
class ListNetLoss(torch.nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, scores, relevance_labels):
        # Convert scores and relevance labels to probability distributions
        # using softmax. This creates the predicted and target ranking distributions.
        relevance_labels = torch.tensor(relevance_labels) # convert into tensor 
        relevance_labels = torch.unsqueeze(relevance_labels, 0) # turn into [B, N] 
        scores = torch.unsqueeze(scores, 0) # turn into [B, N] 
        #print("Truth labels shape after tensor conversion : " + str(relevance_labels.shape))
        #print("Model score shape : " + str(scores.shape))

        log_predicted_probs = F.log_softmax(scores, dim=-1)
        target_probs = relevance_labels / relevance_labels.sum(dim=-1, keepdim=True)

        # Compute KL divergence between the target and predicted distributions.
        # F.kl_div expects log-probabilities for the input, so we apply log here.
        # The reduction='batchmean' averages the loss across the batch.
        loss = F.kl_div(log_predicted_probs, target_probs, reduction='batchmean')

        return loss