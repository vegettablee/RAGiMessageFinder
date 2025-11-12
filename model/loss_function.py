# use binary cross entropy as the loss function, with BCE threshold as a parameter as I will need to experiment with this 
def compute_loss_f1(predicted, all_node_choices, correct_nodes): 
  num_predictions = len(predicted) 
  TP = 0 
  FP = 0
  FN = 0 
  print(type(correct_nodes[0]))
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

def compute_loss_BCE(): 
  # might use this later if f1 score doesn't work well 
  print("HI") 