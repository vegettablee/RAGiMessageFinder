# Conversation Disentanglement RL Model

A machine learning approach to separating interleaved conversation threads in multi-party chat data using a neural architecture with self-attention and teacher forcing.

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

The model uses a **dual-head architecture with self-attention** that both ranks messages and decides which to keep:

```
Input: Message Embeddings [batch_size, num_messages, 768]
   ↓
Self-Attention (4 heads):
   └─ MultiheadAttention(embed_dim=768, num_heads=4)
   └─ Residual Connection: x = x + attention_output
   ↓
Shared Backbone:
   ├─ Linear(768 → 256) + ReLU + Dropout(0.2)
   ├─ Linear(256 → 128) + ReLU + Dropout(0.2)
   └─ Linear(128 → 64)  + ReLU + Dropout(0.2)
   ↓
Dual Output Heads:
   ├─ Rank Head:  Linear(64 → 1) → [B, N]  (ranks messages by thread membership)
   └─ Keep Head:  Linear(64 → 1) → [B, N]  (decides which messages to keep/discard)
```

**Key Design Choices:**

1. **Self-Attention Layer**:
   - **4 attention heads** learn different message relationships:
     - Temporal proximity (messages close in time)
     - Semantic similarity (similar content)
     - Speaker patterns (same author interactions)
     - Reply-to relationships (conversational flow)
   - **Residual connection** preserves original embeddings while adding contextual information
   - Allows each message to "attend to" all other messages in the conversation

2. **Message Embeddings**:
   - Uses `SentenceTransformer("all-mpnet-base-v2")`
   - 768-dimensional dense representations
   - Captures semantic meaning of each message

3. **Fully Connected Backbone**:
   - Progressive dimension reduction: 768 → 256 → 128 → 64
   - ReLU activation for non-linearity
   - 20% dropout for regularization

4. **Dual Output Heads**:
   - **Rank Head**: Predicts ranking scores for messages using ListNetLoss
     - Top-k selection based on highest softmax scores
     - Learns relative importance and ordering of messages
   - **Keep Head**: Predicts binary keep/discard decisions using BCELoss
     - Sigmoid activation with temperature scaling (T=0.5 for sharper boundaries)
     - Threshold at 0.50 for final keep/discard decisions
   - **Joint Loss**: Weighted sum combines both heads (rank_weight=1.0, keep_weight=1.25)
     - Allows both heads to directly influence each other during training

## Training Strategy

### Iterative Thread Prediction with Teacher Forcing

The model is trained using a **sequential teacher-forcing approach with dual head predictions**:

```python
For each conversation:
    remaining_messages = all_messages

    For each ground_truth_thread:
        # 1. Encode remaining messages
        embeddings = encode(remaining_messages)

        # 2. Forward pass (with self-attention + dual heads)
        attn_output = self_attention(embeddings)
        embeddings = embeddings + attn_output  # Residual
        rank_logits, keep_logits = model(embeddings)

        # 3a. Rank Head: Select top-k messages by ranking scores
        rank_probs = softmax(rank_logits)
        rank_predictions = topk(rank_probs, k=len(ground_truth_thread))

        # 3b. Keep Head: Select messages above sigmoid threshold
        keep_probs = sigmoid(keep_logits / temperature)
        keep_predictions = keep_probs > threshold  # T=0.5, threshold=0.50

        # 4. Compute joint loss from both heads
        rank_labels = rank_based_labels(ground_truth_thread)
        keep_labels = binary_labels(ground_truth_thread)

        rank_loss = ListNetLoss(rank_logits, rank_labels)
        keep_loss = BCEWithLogitsLoss(keep_logits, keep_labels)
        total_loss += rank_weight * rank_loss + keep_weight * keep_loss

        # 5. Teacher forcing: remove ground truth messages
        remaining_messages = remaining_messages - ground_truth_thread

    # 6. Backpropagate accumulated loss
    total_loss.backward()
    optimizer.step()
```

### Teacher Forcing Benefits

**Teacher forcing** guides the model by:
- Showing it the correct path through the conversation
- Preventing error accumulation early in training
- Ensuring each prediction operates on a "clean slate"

### Loss Function

**Joint Loss Function**: Weighted sum of two complementary losses
```python
# Rank Head: ListNetLoss for ranking messages
rank_labels = descending_ranks(ground_truth_thread)  # [3, 2, 1, 0, 0, ...]
rank_loss = ListNetLoss(rank_logits, rank_labels)

# Keep Head: BCEWithLogitsLoss for binary keep/discard decisions
keep_labels = binary_labels(ground_truth_thread)     # [1, 1, 1, 0, 0, ...]
keep_loss = BCEWithLogitsLoss(keep_logits, keep_labels)

# Joint Loss: Weighted combination (EXPERIMENTAL)
total_loss = rank_weight * rank_loss + keep_weight * keep_loss
# Currently experimenting with: rank_weight=1.0, keep_weight=1.25
# Previous trials: keep_weight=[1.5, 2.0, 1.25]
```

**Why Dual Head Loss?**
- **Rank Head (ListNetLoss)**:
  - Captures **ordering information** (not just set membership)
  - Better for conversational flow where message sequence matters
  - Uses cross-entropy between probability distributions
- **Keep Head (BCELoss)**:
  - Provides explicit binary signal for message selection
  - **Temperature scaling (EXPERIMENTAL)**: T=0.5 creates sharper decision boundaries
    - Previous temperature trials: [0.75, 0.5]
  - Threshold=0.50 for final keep/discard decisions
  - Allows model to learn when to keep/discard messages independently
- **Joint Training**:
  - Both heads influence each other during backpropagation
  - Rank head learns "which and in what order", keep head learns "whether to include"
  - **Weighted loss experiments**: Testing different weight ratios to balance head contributions
    - Higher keep_weight compensates for keep head's lower initial accuracy

**Evaluation Metrics** (tracked separately for each head + combined):
- **F1 Score**: Harmonic mean of precision and recall (primary metric)
- **Accuracy**: Correct predictions with exact positional matching
- **Precision**: Proportion of predicted messages that are correct
- **Recall**: Proportion of ground truth messages that are predicted
- **Specificity**: Proportion of non-thread messages correctly excluded

## Results

### Architecture Evolution Results

Training on IRC Disentanglement Dataset (max 17 messages per example)

**Single Head Without Attention** (10 epochs)
| Metric | F1 Score | Accuracy | Precision | Recall | Specificity | Loss |
|--------|----------|----------|-----------|--------|-------------|------|
| Result | 0.687 | 0.449 | 0.687 | 0.687 | 0.692 | 0.750 |

**Single Head + 4-Head Self-Attention** (25 epochs)
| Metric | F1 Score | Accuracy | Precision | Recall | Specificity | Loss |
|--------|----------|----------|-----------|--------|-------------|------|
| Result | 0.784 | 0.518 | 0.784 | 0.783 | 0.761 | 0.586 |

**Dual Head + 4-Head Self-Attention** (30 epochs, current approach)

*Combined Metrics (average of both heads):*
| Metric | F1 Score | Accuracy | Precision | Recall | Specificity | Loss |
|--------|----------|----------|-----------|--------|-------------|------|
| Result | 0.679 | 0.567 | 0.692 | 0.704 | 0.799 | 1.044 |

*Rank Head Performance:*
| Metric | F1 Score | Accuracy | Precision | Recall | Specificity |
|--------|----------|----------|-----------|--------|-------------|
| Result | 0.809 | 0.534 | 0.809 | 0.809 | 0.804 |

*Keep Head Performance:*
| Metric | F1 Score | Accuracy | Precision | Recall | Specificity |
|--------|----------|----------|-----------|--------|-------------|
| Result | 0.549 | 0.599 | 0.575 | 0.599 | 0.794 |

**Key Observations:**
1. **Self-attention integration**: Added multi-head attention to capture message relationships
2. **Dual head approach**: Rank head shows strong performance (F1: 0.809), while keep head is still being optimized
3. **Ongoing experiments**: Testing different temperature values (0.5, 0.75) and loss weights (1.0/1.25, 1.0/1.5, 1.0/2.0) to improve keep head accuracy
4. **Higher specificity**: Dual head architecture achieves 0.799 combined specificity, showing better non-thread message exclusion

### Dataset Characteristics

**Thread Distribution:**
- ~42.7% lone threads (single messages)
- ~2.7% long threads (>8 messages)
- ~54.6% standard threads (2-8 messages)

**Pruning Strategy:**
- 75% of lone threads removed (reduce class imbalance)
- 25% of long threads removed (reduce outliers)
- Improves training focus on typical conversation patterns

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
counter > 1:  Referenced by multiple threads (decrement when used)
counter = 1:  Self-reference (thread starter or single use)
counter = 0:  Leaf node (ready for removal)
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
4. Handling branching (one message replied to by multiple messages)

## Training Configuration

```python
# Model hyperparameters
INPUT_DIM = 768          # Embedding dimension
ATTENTION_HEADS = 4      # Multi-head self-attention
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.2
ACTIVATION = ReLU
OUTPUT_HEADS = 2         # Rank head + Keep head

# Training hyperparameters
EPOCHS = 30
BATCH_SIZE = 5
LEARNING_RATE = 0.001
OPTIMIZER = Adam

# Loss function hyperparameters (EXPERIMENTAL)
RANK_HEAD_WEIGHT = 1.0          # Weight for ListNetLoss
KEEP_HEAD_WEIGHT = 1.25         # Weight for BCELoss (testing: 1.25, 1.5, 2.0)
TEMPERATURE = 0.5               # Temperature scaling for keep head (testing: 0.5, 0.75)
SIGMOID_THRESHOLD = 0.50        # Threshold for binary keep/discard decisions

# Data configuration
MAX_MESSAGES_PER_EXAMPLE = 17   # Truncate long conversations
NUM_EXAMPLES = 5000             # Total dataset size
TEACHER_FORCE = True

# Pruning configuration
PRUNE_SELF_NODES_PROB = 0.75   # Remove 75% of lone threads
PRUNE_LONG_NODES_PROB = 0.25   # Remove 25% of threads >8 messages
PRUNE_LONG_NODE_LEN = 8        # Threshold for "long" threads
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

## Architecture Evolution

### v1.0: Single-Head Baseline
- **Architecture**: Single thread prediction head
- **Loss**: Binary Cross-Entropy (BCE)
- **Results**: F1: 0.687, Accuracy: 0.449

### v2.0: Single-Head + Self-Attention
- **Architecture**: Self-attention (4 heads) + single thread prediction head
- **Loss**: ListNetLoss (ranking-aware)
- **Key improvements**:
  - Self-attention learns message relationships
  - Ranking-aware loss for conversation flow
- **Results**: F1: 0.784, Accuracy: 0.518

### v3.0: Dual-Head + Self-Attention (Current)
- **Architecture**: Self-attention (4 heads) + two output heads (rank + keep/discard)
- **Loss**: Joint loss with ListNetLoss + BCELoss
- **Key improvements**:
  - Rank head: Learns message ordering (F1: 0.809)
  - Keep head: Learns binary keep/discard decisions (F1: 0.549)
  - Joint loss allows both heads to influence each other
  - Experimenting with temperature scaling and loss weights for optimization
- **Results**: Rank Head F1: 0.809, Keep Head F1: 0.549 (in progress)

## Key Implementation Details

### 1. Self-Attention Mechanism

**Why Self-Attention?**
- Messages in conversations have **long-range dependencies**
- Reply patterns aren't always sequential (message 5 might reply to message 2)
- Multiple speakers create complex interaction graphs
- Attention lets model learn "which messages relate to each other"

**Implementation**:
```python
# 4-head multi-head attention
attn_output, _ = self.attn(x, x, x)  # Q, K, V all from same input
x = x + attn_output  # Residual connection preserves original info
```

**What the attention heads learn** (empirical observation):
- Head 1: Same-speaker patterns
- Head 2: Temporal proximity
- Head 3: Semantic similarity (topic coherence)
- Head 4: Reply-to relationships

### 2. Iterative Thread Prediction

Predicting threads **one at a time** (instead of all at once):
- Simpler learning problem (binary: in thread or not)
- Naturally handles variable number of threads
- Allows teacher forcing for each thread
- Matches inference-time behavior

### 3. Dual Head Architecture with Joint Loss

**Rank Head (ListNetLoss)** + **Keep Head (BCELoss)**:
- **Rank Head**: "Rank all messages by thread membership" (1st, 2nd, 3rd...)
  - Captures **ordering** which is crucial for conversational coherence
  - Consistently achieves strong F1 scores (0.809)
- **Keep Head**: "Should this message be kept or discarded?" (binary decision)
  - Provides explicit signal for message selection
  - Temperature scaling creates sharper decision boundaries
  - Currently being optimized through weight and temperature experiments
- **Joint Loss**: Weighted combination allows both heads to inform each other during training
  - Experimenting with different weight ratios (1.0/1.25, 1.0/1.5, 1.0/2.0)
  - Higher keep_weight compensates for keep head's initially lower accuracy

### 4. Dynamic Pruning

**Class Imbalance Problem**:
- ~43% of threads are lone messages (noise)
- ~3% of threads are very long (outliers)

**Solution**:
- Randomly prune 75% of lone threads during training
- Randomly prune 25% of ultra-long threads
- Forces model to focus on typical conversation patterns

## Project Structure

```
model/
├── CD_model.py           # Neural architecture with self-attention
├── training_loop.py      # Training loop with teacher forcing
├── loss_function.py      # ListNetLoss + evaluation metrics
├── dataset_utils.py      # IRC dataset loading
├── construct_tree.py     # Message tree construction
├── debug_display.py      # Visualization tools
└── results.json          # Training metrics history

dataset/
└── irc-disentanglement/  # IRC dataset
    └── data/
        ├── train/
        ├── dev/
        └── test/

proj_docs/
└── proj_doc.md          # Detailed development notes
```

## Dependencies

```
torch>=2.0.0
sentence-transformers>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## Future Directions

1. **Hybrid Learning**: Once F1 > 0.80, switch from teacher forcing to model's own predictions (reinforcement learning-style)
2. **Additional Features**: Speaker embeddings, temporal features, explicit reply-to detection
3. **Attention Visualization**: Analyze what patterns each attention head learns
4. **Model Compression**: Distill to smaller model for deployment
5. **Multi-Dataset Evaluation**: Test on Slack, Discord, Reddit conversation data

## Citation

If you use this work, please cite:
```
@misc{conversation-disentanglement-2025,
  author = {Preston Rank},
  title = {Conversation Disentanglement with Self-Attention Neural Networks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/prestonrank/RAGMessages}
}
```

## License

MIT License - See LICENSE file for details
