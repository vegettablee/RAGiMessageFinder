# Conversation Disentanglement with Neural Networks

A machine learning approach to separating interleaved conversation threads in multi-party chat data using a novel dual-head neural architecture with teacher forcing.

## Problem Statement

In multi-party conversations (IRC, Slack, Discord), multiple conversation threads happen simultaneously. This project trains a neural network to identify which messages belong together, even when they're interleaved.

**Example Input:**
```
[14:32] <BurgerMann> hey how do I install nvidia drivers?
[14:33] <delire> BurgerMann: sudo apt-get install nvidia-drivers
[14:33] <Seveas> anyone know if mysql is down?
[14:34] <BurgerMann> delire: which version should I use?
[14:34] <delire> Seveas: works for me
```

**Model Output:**
- Thread 1: Messages 1, 2, 4 (nvidia discussion)
- Thread 2: Messages 3, 5 (mysql discussion)

## Architecture

### Neural Network Design

The model uses a **dual-head feedforward architecture** with message embeddings as input:

```
Input: Message Embeddings [batch_size, num_messages, 768]
   ↓
Shared Backbone:
   ├─ Linear(768 → 256) + ReLU + Dropout(0.2)
   ├─ Linear(256 → 128) + ReLU + Dropout(0.2)
   └─ Linear(128 → 64)  + ReLU + Dropout(0.2)
   ↓
Dual Output Heads:
   ├─ Thread Head:  Linear(64 → 1) → [B, N]  (which messages belong to current thread)
   └─ Keep Head:    Linear(64 → 1) → [B, N]  (which messages to keep for next iteration)
```

**Key Design Choices:**

1. **Dual-Head Architecture**:
   - **Thread Head**: Predicts probability that each message belongs to the current thread
   - **Keep Head**: Predicts which messages to retain for subsequent thread predictions

2. **Message Embeddings**:
   - Uses `SentenceTransformer("all-mpnet-base-v2")`
   - 768-dimensional dense representations
   - Captures semantic meaning of each message

3. **Fully Connected Backbone**:
   - Progressive dimension reduction: 768 → 256 → 128 → 64
   - ReLU activation for non-linearity
   - 20% dropout for regularization

## Training Strategy

### Iterative Thread Prediction with Teacher Forcing

The model is trained using a **sequential teacher-forcing approach** where it learns to predict threads one at a time:

```python
For each conversation:
    remaining_messages = all_messages

    For each ground_truth_thread:
        # 1. Encode remaining messages
        embeddings = encode(remaining_messages)

        # 2. Forward pass
        thread_probs, keep_probs = model(embeddings)

        # 3. Select top-k messages (k = length of ground truth)
        predictions = topk(thread_probs, k=len(ground_truth_thread))

        # 4. Compute loss
        loss = BCE(predictions, ground_truth_labels)

        # 5. Teacher forcing: remove ground truth messages
        remaining_messages = remaining_messages - ground_truth_thread

    # 6. Backpropagate accumulated loss
    loss.backward()
    optimizer.step()
```

### Teacher forcing 

**Teacher forcing** guides the model by:
- Showing it the correct path through the conversation
- Preventing error accumulation early in training
- Ensuring each prediction operates on a "clean slate"

**Future Plan**: Switch to **hybrid learning** once F1 > 0.75:
- Let the model make its own choices (remove predicted messages, not ground truth)
- Acts like reinforcement learning without explicit rewards
- Encourages robust predictions when graph structure is uncertain

### Loss Function

**Primary Loss**: Binary Cross-Entropy (BCE)
```python
# For each message, predict: in thread (1) or not in thread (0)
loss = BCEWithLogitsLoss(thread_logits, binary_labels)
```

**Evaluation Metrics**:
- **F1 Score**: Harmonic mean of precision and recall (primary metric)
- **Accuracy**: Intersection over union of predicted vs. ground truth
- **Per-thread metrics**: Tracked for each conversation thread

## Message Tree Representation

### Graph Construction

Messages are represented as a **fully-connected directed graph**:

```
msg1 → [msg2, msg3, msg4, ...]  (connects to all future messages)
msg2 → [msg3, msg4, ...]        (connects to all future messages)
msg3 → [msg4, ...]              (connects to all future messages)
msg4 → []                        (leaf node)
```

### Node Counter System

Each node has a **reference counter** tracking how many times it appears in ground truth threads:

```python
counter > 0:  Referenced by multiple threads (decrement when used)
counter = 1:  Self-reference (thread starter)
counter = 0:  Leaf node (never referenced)
```

**Dynamic Removal**:
- After each thread prediction, decrement counters for used nodes
- Remove nodes when counter reaches 0
- Prevents the model from re-using already-assigned messages

### Thread Extraction

Ground truth threads are extracted by:
1. Building forward reference graph from annotations
2. Finding root nodes (messages that start threads)
3. Traversing graph to extract linear conversation paths
4. Handling branching (one message replies to by multiple messages)

## Training Configuration

```python
# Model hyperparameters
INPUT_DIM = 768          # Embedding dimension
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.2
ACTIVATION = ReLU

# Training hyperparameters
EPOCHS = 1
BATCH_SIZE = 5
LEARNING_RATE = 0.001
OPTIMIZER = Adam

# Data configuration
MAX_MESSAGES_PER_EXAMPLE = 20  # Truncate long conversations
NUM_EXAMPLES = 1000
TEACHER_FORCE = True

# Prediction thresholds
F1_THRESHOLD = 0.5       # Target for switching to hybrid learning
KEEP_THRESHOLD = 0.75    # Probability threshold for keep/discard head
```

## Dataset

### IRC Disentanglement Dataset

**Source**: Human-annotated IRC chat logs with conversation thread labels

**Format**:
```python
{
  'id': [1, 2, 3, 4, ...],
  'raw': ["[09:15] <alice> message text", ...],
  'date': ['2004-12-25', ...],
  'connections': [
    [1],    # msg 1 is thread root (self-reference)
    [1],    # msg 2 replies to msg 1
    [3],    # msg 3 is thread root
    [2],    # msg 4 replies to msg 2
    ...
  ]
}
```

**Splits**:
- `train/`: Training conversations (~800 files)
- `dev/`: Validation set (~200 files)
- `test/`: Final evaluation (~200 files)

**Location**: `dataset/irc-disentanglement/data/`

## Key Implementation Details

### 1. Dual Output Heads

**Thread Head**:
- Predicts which messages belong to current thread
- Binary classification for each candidate message
- Uses top-k selection based on ground truth length

**Keep Head**:
- Manages message tree during training
- Prevents re-using messages across multiple threads
- Learns which messages are "consumed" vs. "reusable"

### 2. Iterative Prediction

Predicting threads **one at a time** (instead of all at once):
- Simpler learning problem (binary: in thread or not)
- Naturally handles variable number of threads
- Allows teacher forcing for each thread
- Matches inference-time behavior

### 3. Fully-Connected Graph

Initially, each message can connect to **any future message**:
- Models uncertainty about conversation structure
- Lets model learn which connections are meaningful
- Pruned dynamically as threads are predicted
- Computationally tractable (max 20 messages per example)

## Project Structure

```
model/
├── CD_model.py
├── training_loop.py
├── loss_function.py
├── dataset_utils.py
├── construct_tree.py
├── debug_display.py
└── results/

dataset/
└── irc-disentanglement/
```

## Dependencies

```
torch
sentence-transformers
numpy
scikit-learn
```
