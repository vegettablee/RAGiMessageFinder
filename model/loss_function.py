import torch
import torch.nn.functional as F

# use binary cross entropy as the loss function, with BCE threshold as a parameter as I will need to experiment with this 
def compute_loss_f1(predicted, all_node_choices, correct_nodes): 
  num_predictions = len(predicted) 
  TP = 0 
  FP = 0
  FN = 0 
  correct_set = set(correct_nodes)
  predicted_set = set(predicted)
  for n in predicted: 
    if n in correct_set: 
      TP += 1
    else: 
      FP += 1
    
  for n in correct_nodes: 
    if n not in predicted: 
      FN += 1
  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  recall = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  return f1_score

class ListNetLoss(torch.nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, scores, relevance_labels):
        """
        Computes the ListNet loss using KL divergence.

        Args:
            scores (torch.Tensor): Predicted scores for documents in a query.
                                   Shape: (batch_size, list_size)
            relevance_labels (torch.Tensor): Ground-truth relevance labels for documents.
                                             Shape: (batch_size, list_size)
        Returns:
            torch.Tensor: The computed ListNet loss.
        """
        # Convert scores and relevance labels to probability distributions
        # using softmax. This creates the predicted and target ranking distributions.
        predicted_probs = F.softmax(scores, dim=-1)
        target_probs = F.softmax(relevance_labels, dim=-1)

        # Compute KL divergence between the target and predicted distributions.
        # F.kl_div expects log-probabilities for the input, so we apply log here.
        # The reduction='batchmean' averages the loss across the batch.
        loss = F.kl_div(predicted_probs.log(), target_probs, reduction='batchmean')

        return loss