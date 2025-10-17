import torch
import torch.nn.functional as F

# Similarity thresholds
BURST_SIMILARITY_THRESHOLD = 0.32
TEXT_SIMILARITY_THRESHOLD = 0.22
THREAD_TEXT_SIMILARITY_THRESHOLD = 0.26


def compute_burst_similarity(burst1, burst2, model, burst1_idx, burst2_idx):

  first = ",".join([item[3] for item in burst1])
  second = ",".join([item[3] for item in burst2])

  first_emb = model.encode(first, convert_to_tensor=True)
  second_emb = model.encode(second, convert_to_tensor=True)

  similarity = F.cosine_similarity(first_emb.unsqueeze(0), second_emb.unsqueeze(0))

  if similarity.item() > BURST_SIMILARITY_THRESHOLD:
    print(f"Bursts {burst1_idx} and {burst2_idx} are similar, keeping them together. Score {similarity.item()}")
    return True
  else:
    print(f"Bursts {burst1_idx} and {burst2_idx} are not similar, keeping them separate. Score {similarity.item()}")
    return False


def compute_text_similarity(message1, message2, model, similarity_threshold=TEXT_SIMILARITY_THRESHOLD):
  
  print(f"Message 1 : {message1}")
  print(f"Message 2 : {message2}")

  first_emb = model.encode(message1, convert_to_tensor=True)  # 1D shape
  second_emb = model.encode(message2, convert_to_tensor=True)

  similarity = F.cosine_similarity(first_emb.unsqueeze(0), second_emb.unsqueeze(0))

  if similarity.item() > similarity_threshold:
    print(f"Messages were similar. Score {similarity.item()}")
    return True, similarity.item()
  else:
    print(f"Messages were not similar. Score {similarity.item()}")
    return False, similarity.item()


def compute_thread_similarity(burst1, burst2, model): 
  print("")