# RAGMessages - Conversation Disentanglement Project

## Project Overview

RAGMessages is a machine learning project focused on **conversation disentanglement** - the task of separating interleaved conversation threads from multi-party chat data. The project uses the IRC Disentanglement dataset to train a neural network that can identify which messages belong to the same conversation thread, even when multiple conversations are happening simultaneously.

### Current Focus

The project is currently in the **prototype architecture phase**, with emphasis on:
1. Training a multi-head neural network for conversation thread prediction
2. Iterative teacher-forcing approach for learning conversation structure
3. Evaluation using F1 scores and accuracy metrics
4. Building a message tree representation for conversation graphs

**Secondary Components** (future work):
- RAG (Retrieval-Augmented Generation) for message retrieval
- Chunking algorithms for topic shift detection

---

## Core Problem Statement

Given a conversation with interleaved messages like:

```
[14:32] <BurgerMann> hey how do I install nvidia drivers?
[14:33] <delire> BurgerMann: sudo apt-get install nvidia-drivers
[14:33] <Seveas> anyone know if mysql is down?
[14:34] <BurgerMann> delire: which version should I use?
[14:34] <delire> Seveas: works for me
[14:35] <techguy> BurgerMann: try nvidia-driver-470
[14:35] <Seveas> delire: hmm weird, must be my connection
[14:36] <BurgerMann> thanks both!
```

The model should identify distinct threads:
- **Thread 1**: Messages 1, 2, 4, 6, 8 (nvidia drivers discussion)
- **Thread 2**: Messages 3, 5, 7 (mysql connection discussion)

---

## ML Model Architecture

### CDModel (Conversation Disentanglement Model)

Location: `model/CD_model.py`

```
Input: [batch_size, num_candidates, 768]  (message embeddings)
    ↓
Layer 1: Linear(768 → 256) + ReLU + Dropout(0.2)
    ↓
Layer 2: Linear(256 → 128) + ReLU + Dropout(0.2)
    ↓
Layer 3: Linear(128 → 64) + ReLU + Dropout(0.2)
    ↓
Multi-Head Output:
├─→ Thread Logits: Linear(64 → 1) → [batch_size, num_candidates]
└─→ Keep Logits:   Linear(64 → 1) → [batch_size, num_candidates]
```

### Key Architecture Details

**Input Features**:
- Message embeddings from `SentenceTransformer("all-mpnet-base-v2")`
- Dimension: 768 (embedding size)
- Each conversation has variable number of candidate messages

**Two Output Heads**:
1. **Thread Prediction Head** (`thread_logits`):
   - Predicts which messages belong to the current thread
   - Uses binary cross-entropy loss (BCEWithLogitsLoss)
   - Sigmoid activation converts logits to probabilities
   - Top-k selection based on ground truth thread length

2. **Keep/Discard Head** (`keep_logits`):
   - Predicts whether to keep each node for the next iteration
   - Threshold: probability > 0.75 means keep the node
   - Used for managing the message tree during training

**Hyperparameters**:
- Input dimension: 768
- Hidden layers: 256 → 128 → 64
- Dropout rate: 0.2
- Activation: ReLU
- Loss function: BCEWithLogitsLoss

---

## Training Process

### Training Loop Overview

Location: `model/training_loop.py`

The training uses a **teacher-forcing approach** with iterative thread prediction:

```
For each conversation example:
  1. Build message tree from ground truth annotations
  2. For each ground truth thread:
     a. Encode all remaining messages with SentenceTransformer
     b. Forward pass through CDModel → get thread_logits and keep_logits
     c. Apply sigmoid to get probabilities
     d. Select top-k predictions (k = length of ground truth thread)
     e. Apply keep/discard threshold (0.75)
     f. Compute F1 score and accuracy
     g. Calculate BCE loss comparing predictions to ground truth
     h. Remove used nodes from message tree (teacher forcing)
  3. Backpropagate accumulated loss for entire example
  4. Update model weights
```

### Teacher Forcing Strategy

**When `TEACHER_FORCE = True`** (current default):
- After each thread prediction, remove ground truth nodes from the tree
- This guides the model to learn from correct examples
- Ensures training follows the optimal path

**When `TEACHER_FORCE = False`** (future hybrid approach):
- Remove only the nodes that the model predicted
- Let the model make its own choices (reinforcement learning style)
- Plan: Switch to this mode once accuracy reaches ~75%

### Training Configuration

```python
EPOCHS = 1
BATCH_SIZE = 5
F1_THRESHOLD = 0.5
MAX_MESSAGES_PER_EXAMPLE = 20  # Truncate long conversations
NUM_EXAMPLES = 1000
TEACHER_FORCE = True
```

### Loss Function

Location: `model/loss_function.py`

**Primary Loss**: Binary Cross-Entropy (BCE)
```python
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(node_logits, true_labels)
```

**Evaluation Metric**: F1 Score
```python
f1_score = compute_loss_f1(predicted_ids, remaining_nodes, correct_thread)

# Where:
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Accuracy Calculation**:
```python
accuracy = len(predicted ∩ ground_truth) / len(predicted)
```

---

## Dataset Structure

### IRC Disentanglement Dataset

Location: `dataset/irc-disentanglement/`

The dataset contains IRC chat logs with human-annotated conversation threads.

**Dataset Splits**:
- `train/`: Training conversations
- `dev/`: Development/validation set
- `test/`: Test set

**File Structure**:
Each conversation has two files:
- `*.ascii.txt`: Raw IRC messages with timestamps
- `*.annotation.txt`: Ground truth thread connections

### Data Format

Location: `model/dataset_utils.py`

Each example is a dictionary with:

```python
{
  'id': [1, 2, 3, 4, ...],           # Message IDs (1-indexed)
  'raw': [                            # Raw message strings
    "[09:15] <alice> does anyone have experience with docker?",
    "[09:16] <bob> alice: yeah, what's the issue?",
    ...
  ],
  'date': ['2004-12-25', ...],       # Message dates
  'connections': [                    # Ground truth connections
    [1],      # Message 1 references itself (thread start)
    [1],      # Message 2 replies to message 1
    [3],      # Message 3 starts new thread
    [2],      # Message 4 replies to message 2
    ...
  ]
}
```

**Connections Explained**:
- `connections[i]` shows which previous message(s) message `i+1` replies to
- Self-reference `[i+1]` means the message starts a new thread
- Backward references `[j]` where `j < i+1` show thread continuation

### Message Tree Construction

Location: `model/construct_tree.py`

The message tree is built from the ground truth connections:

**Process**:
1. Parse connections to identify forward references (node → future nodes)
2. Build a fully-connected graph where each node connects to all future nodes
3. Each node has a **counter** tracking how many times it's referenced
4. Extract linear conversation threads by traversing the graph
5. Handle branching: when a node has multiple children, create separate threads

**Node Counter Logic**:
- `counter > 0`: Node is referenced by future nodes (counter = number of references)
- `counter = 1`: Self-reference (starts new thread)
- `counter = 0`: Leaf node (not referenced by anyone)

**As threads are predicted**:
- Decrement counter when a node is used
- Remove node when counter reaches 0
- Remove node from other nodes' forward edges

---

## Key Components

### 1. Model Definition
**File**: `model/CD_model.py`

- `CDModel`: Main neural network class
- `forward()`: Forward pass returning two output heads
- `get_model_config()`: Returns model architecture info for logging

### 2. Training Loop
**File**: `model/training_loop.py`

- `training_loop()`: Main training orchestration
- `save_data()`: Saves model checkpoints and metrics to `results.json`
- Handles epoch tracking, batch processing, and metric aggregation

### 3. Dataset Utilities
**File**: `model/dataset_utils.py`

- `load_irc_file()`: Loads single IRC conversation with annotations
- `load_dataset()`: Loads full dataset split (train/dev/test)
- `get_dataloader()`: Creates batching iterator
- Handles ID renumbering and connection mapping

### 4. Message Tree Construction
**File**: `model/construct_tree.py`

- `build_message_tree()`: Constructs fully-connected message graph
- `message_node`: Class representing a single message with forward edges
- `remove_used_nodes()`: Decrements/removes nodes after thread prediction
- Handles root node detection and thread extraction

### 5. Loss Functions
**File**: `model/loss_function.py`

- `compute_loss_f1()`: Calculates F1 score for thread predictions
- `compute_loss_BCE()`: Placeholder for alternative loss (not yet implemented)

### 6. Debugging & Visualization
**File**: `model/debug_display.py`

Provides functions to display:
- Full conversation with message IDs
- Raw connections from dataset
- Ground truth threads
- Node counters and reference counts
- Thread predictions with probabilities
- Embedding information

### 7. Model Checkpointing
**File**: `model/save.py`

Handles saving and loading model checkpoints during training.

---

## Development Workflow

### Running Training

```bash
cd /Users/prestonrank/RAGMessages
python model/training_loop.py
```

This will:
1. Load 1000 examples from the training set
2. Run 1 epoch of training
3. Display per-example metrics (loss, F1, accuracy)
4. Save results to `model/results/results.json`

### Monitoring Progress

During training, you'll see:
```
================================================================================
EPOCH 1/1
================================================================================

=== FORWARD GRAPH DEBUG ===
Node 1 → Future nodes: [2, 4, 8]
Node 3 → Future nodes: [5, 7]
...

Example 1/1000 | Loss: 0.6234 | Avg F1: 0.7145 | Avg Accuracy: 0.6833
Example 2/1000 | Loss: 0.5891 | Avg F1: 0.7423 | Avg Accuracy: 0.7100
...

================================================================================
EPOCH 1 SUMMARY
================================================================================
Average Loss:     0.5234
Average F1 Score: 0.7541
Average Accuracy: 0.7234
================================================================================
```

### Debug Mode

The first N examples show detailed debug information:
- Full conversation display
- Raw connections from annotations
- Ground truth threads
- Node counters
- Embedding dimensions
- Thread predictions with probabilities

Configure in `model/debug_display.py`:
```python
def should_show_debug_for_example(batch_idx):
    return batch_idx < 3  # Show first 3 examples
```

### Evaluation Metrics

**Tracked Metrics**:
1. **Loss**: BCEWithLogitsLoss per example/epoch
2. **F1 Score**: Harmonic mean of precision and recall
3. **Accuracy**: Overlap between predicted and ground truth threads

**Saved to**: `model/results/results.json`

```json
{
  "timestamp": "2024-11-14T...",
  "epoch": 1,
  "metrics": {
    "avg_loss": 0.5234,
    "avg_f1_score": 0.7541,
    "avg_accuracy": 0.7234
  },
  "model_config": {...},
  "training_config": {...}
}
```

---

## Algorithm Documentation

Location: `algorithm_docs/`

### 1. Baseline Implementation
**File**: `algorithm_docs/baseline.md`

Describes the **Time-Gapped Hierarchical Agglomerative Clustering (HAC)** baseline:

**Two-Stage Approach**:
1. **Stage 1 - Temporal Chunking**:
   - Split messages on time gaps > 5 minutes
   - Creates manageable chunks for processing

2. **Stage 2 - Semantic Clustering**:
   - Use sentence embeddings (all-MiniLM-L6-v2)
   - Apply HAC with average linkage
   - Distance threshold: 0.5

**Expected Performance**:
- Accuracy: 72-76%
- F1 Score: 0.72-0.76

**Baseline Purpose**: The ML model needs to beat this by 8-10% (target: 80-85% accuracy) to justify the added complexity.

### 2. Algorithm Experiments
**File**: `algorithm_docs/algorithm_doc.md`

Chronicles the evolution of chunking approaches:

1. **Burst-Based Chunking**:
   - Group messages within 5-minute windows (bursts)
   - Group bursts within 45-minute windows (sections)
   - Use cosine similarity for topic shift detection

2. **Micro-Thread Extraction**:
   - Compare messages across adjacent bursts
   - Track conversation micro-threads within sections
   - Apply time decay for similarity scoring

3. **Pivot to ML Approach**:
   - Transition from rule-based to learned model
   - Use IRC dataset for quantifiable metrics
   - Implement teacher-forcing with reinforcement learning

### 3. Project Documentation
**File**: `algorithm_docs/proj_doc.md`

High-level project journey and terminology:
- **Bursts**: Groups within 5 minutes
- **Chunks**: Bursts within 30 minutes or semantically similar
- **Micro-threads**: Individual threads within chunks

---

## Prototype Architecture Evolution

### Initial Approach (Abandoned)
- Reward-based reinforcement learning
- Global sum of cosine similarities as reward
- **Problem**: Reward functions were too vague and sensitive

### Current Approach (Prototype v1)
- Teacher-forcing with supervised learning
- Messages modeled as decision tree nodes
- Two-head architecture for thread + keep/discard prediction
- Ground truth threads sorted in order
- Model predicts threads iteratively

**Key Insight**: When accuracy ratio ≥ 0.75, switch from teacher-forcing to free prediction (hybrid learning approach)

### Future Enhancements

**Input Features** (when simple approach plateaus):
- Message embeddings (current)
- Embeddings of existing threads (average)
- Cosine similarity between message and each thread
- Number of messages in each thread
- Time gap to most recent message in thread
- Speaker patterns and discourse markers

**Model Improvements**:
- Attention mechanisms for message relationships
- Random masking during training
- Better handling of discourse markers ("also", "btw")
- Question-answer structure modeling
- Pronoun resolution and entity tracking

---

## RAG System (Secondary Focus)

Location: `rag.py`

The RAG (Retrieval-Augmented Generation) system is a **future component** for retrieving relevant message chunks.

### Components

**Embedding Model**:
```python
SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
# Dimension: 384
# Similarity: Cosine
```

**Vector Store**: FAISS (Facebook AI Similarity Search)
```python
index = faiss.IndexFlatL2(384)  # L2 distance for retrieval
```

**Workflow**:
1. Load message data from files
2. Chunk messages into groups (6 sentences per chunk)
3. Embed chunks with SentenceTransformer
4. Store embeddings in FAISS index
5. Query with natural language → retrieve similar chunks

**Current Status**: Basic implementation complete, but **not integrated** with conversation disentanglement model. Will be added once thread prediction reaches target accuracy.

---

## Chunking Algorithms (Secondary Focus)

Location: `algorithm/`

The chunking algorithms were initial experiments for topic segmentation. They are **not currently used in training** but may be reintroduced later.

### Available Algorithms

1. **Topic Shift Detection** (`algorithm/topic_shift.py`):
   - Time-based bursting (5-minute threshold)
   - Section grouping (30-minute threshold)
   - Cosine similarity for semantic boundary detection

2. **Micro-Thread Extraction** (`algorithm/micro_thread.py`):
   - Cross-burst message comparison
   - Thread similarity scoring with time decay
   - Isolated thread merging

3. **Similarity Utilities** (`algorithm/similarity.py`):
   - Cosine similarity computation
   - Embedding-based text comparison

### Parameters from Experiments

**Tested Thresholds**:
- Text similarity: 0.15, 0.22 (best: 0.22)
- Burst similarity: 0.35, 0.4, 0.5 (best: 0.35)
- Thread similarity: 0.3

**Time Thresholds**:
- LINE_BURST_TIME: 5 minutes
- MAX_SECTION_TIME: 45 minutes (initial), 30 minutes (current)
- Time decay starts after: 15 minutes

**Statistical Analysis** (from real data):
- 75th percentile gap: 12.67 minutes
- 90th percentile gap: 91.61 minutes
- Largest jump: between 75th and 90th percentile
- Conclusion: Use 15-minute decay threshold

---

## File Structure

```
RAGMessages/
├── model/                          # Core ML components
│   ├── CD_model.py                 # Neural network architecture
│   ├── training_loop.py            # Training orchestration
│   ├── loss_function.py            # F1 score and BCE loss
│   ├── dataset_utils.py            # IRC dataset loading
│   ├── construct_tree.py           # Message tree construction
│   ├── debug_display.py            # Visualization utilities
│   ├── save.py                     # Model checkpointing
│   ├── baseline.py                 # Baseline implementation
│   └── results/
│       ├── results.json            # Training metrics
│       └── results.py              # Results analysis
│
├── algorithm_docs/                 # Documentation
│   ├── proj_doc.md                 # Project journey
│   ├── algorithm_doc.md            # Algorithm evolution
│   └── baseline.md                 # Baseline description
│
├── algorithm/                      # Chunking algorithms (secondary)
│   ├── chunking.py
│   ├── similarity.py
│   ├── micro_thread.py
│   ├── topic_shift.py
│   └── remainder.py
│
├── dataset/                        # IRC disentanglement data
│   └── irc-disentanglement/
│       ├── data/
│       │   ├── train/              # Training conversations
│       │   ├── dev/                # Validation set
│       │   └── test/               # Test set
│       └── tools/                  # Evaluation scripts
│
├── rag.py                          # RAG system (future)
├── app.py                          # Chainlit UI (future)
├── main.ipynb                      # Jupyter experiments
├── requirements.txt                # Python dependencies
└── CLAUDE.md                       # This file
```

---

## Current Status & Next Steps

### Current Status (Prototype Phase 1)

✅ **Completed**:
- CDModel architecture implemented
- Training loop with teacher forcing
- IRC dataset integration
- Message tree construction
- F1 score and accuracy tracking
- Debug visualization tools
- Model checkpointing

🚧 **In Progress**:
- Training on full dataset (currently: 1000 examples, 1 epoch)
- Hyperparameter tuning
- Baseline comparison

### Immediate Next Steps

1. **Complete Initial Training**:
   - Run multiple epochs (10-20)
   - Track convergence
   - Save best model checkpoint

2. **Evaluate Against Baseline**:
   - Implement baseline HAC approach
   - Compare F1 scores on test set
   - Target: Beat 72-76% baseline by 8-10%

3. **Hyperparameter Tuning**:
   - Experiment with hidden layer sizes
   - Adjust dropout rates
   - Try different learning rates
   - Optimize keep/discard threshold (currently 0.75)

4. **Switch to Hybrid Learning**:
   - When F1 > 0.75, disable teacher forcing
   - Let model predict freely (reinforcement learning)
   - Compare hybrid vs. pure supervised performance

### Medium-Term Goals

1. **Enhance Input Features**:
   - Add thread-level embeddings
   - Include cosine similarity features
   - Incorporate time gap features
   - Add speaker pattern features

2. **Model Architecture Improvements**:
   - Experiment with attention mechanisms
   - Try transformer-based encoders
   - Add recurrent layers (LSTM/GRU)
   - Implement graph neural networks

3. **Advanced Evaluation**:
   - Variation of Information (VI) metric
   - Edit distance to ground truth
   - One-to-one overlap scores
   - Per-thread accuracy breakdown

### Long-Term Vision

1. **RAG Integration**:
   - Combine thread prediction with retrieval
   - Use disentangled threads for better RAG chunks
   - Build conversational QA system

2. **Production Deployment**:
   - Optimize inference speed
   - Deploy as API service
   - Build UI with Chainlit (app.py)
   - Handle real-time chat streams

3. **Generalization**:
   - Test on other chat platforms (Slack, Discord, SMS)
   - Transfer learning to new domains
   - Multi-language support

---

## Key Research Questions

### Model Design
- [ ] What input features are most important for thread prediction?
- [ ] Does attention improve performance significantly?
- [ ] Should we use graph neural networks instead of feedforward?
- [ ] How to handle very long conversations (>100 messages)?

### Training Strategy
- [ ] When is the optimal time to switch from teacher forcing to free prediction?
- [ ] Does random masking improve generalization?
- [ ] Should we pre-train on a related task?
- [ ] How to balance thread prediction vs. keep/discard head?

### Evaluation
- [ ] Which metric best reflects conversation quality: F1, VI, or edit distance?
- [ ] How to evaluate partial thread correctness?
- [ ] What baseline is most appropriate for this task?

### Domain Adaptation
- [ ] How well does IRC model transfer to other chat platforms?
- [ ] Can we use discourse markers as explicit features?
- [ ] How to handle speaker attribution in anonymous chats?

---

## Dependencies

See `requirements.txt`:

```
sentence-transformers>=2.2.0    # Message embeddings
openai>=1.0.0                   # Future LLM integration
numpy>=1.24.0                   # Numerical operations
scikit-learn>=1.3.0             # Baseline clustering
pandas>=2.0.0                   # Data manipulation
jupyter>=1.0.0                  # Experimentation
python-dotenv>=1.0.0            # Environment variables
faiss-cpu>=1.7.4                # Vector search (RAG)
tiktoken>=0.5.0                 # Token counting
```

**PyTorch**: Required but not in requirements.txt (install separately based on your system)

---

## Getting Started for Claude Code

### Understanding the Codebase

1. **Start with the architecture docs**:
   - Read `algorithm_docs/proj_doc.md` for project evolution
   - Review `algorithm_docs/baseline.md` to understand the target

2. **Examine the model**:
   - `model/CD_model.py`: Understand the two-head architecture
   - `model/training_loop.py`: See how training works

3. **Understand the data**:
   - `model/dataset_utils.py`: How IRC data is loaded
   - `model/construct_tree.py`: How message graphs are built

### Making Changes

**To modify the model architecture**:
- Edit `model/CD_model.py`
- Update `training_loop.py` if input/output shapes change
- Adjust loss computation if needed

**To change training behavior**:
- Edit `model/training_loop.py`
- Modify hyperparameters at top of file
- Change teacher forcing logic in the loop

**To add new features**:
- Update `construct_tree.py` to extract features
- Modify `CDModel` input dimension
- Update embedding creation in `training_loop.py`

**To experiment with loss functions**:
- Edit `model/loss_function.py`
- Update `training_loop.py` to use new loss
- Track new metrics in the training loop

### Common Tasks

**Run training**:
```bash
python model/training_loop.py
```

**Analyze results**:
```bash
cat model/results/results.json
```

**Test on a single example**:
```python
from model.dataset_utils import get_example
from model.construct_tree import build_message_tree

example = get_example()
tree, threads = build_message_tree(example)
```

**Visualize debug output**:
- Modify `model/debug_display.py` to show more/fewer examples
- Debug output shows during training automatically

---

## Terminology Reference

**Node**: A single message in the conversation, referenced by ID (1-indexed)

**Thread**: A sequence of messages forming a single conversation topic

**Ground Truth Thread**: The correct thread from human annotations (e.g., `(1, 2, 4, 6)`)

**Message Tree**: Fully-connected graph where each node connects to all future nodes

**Counter**: Number of times a node is referenced in ground truth threads

**Forward Edges**: Connections from a node to all messages that come after it

**Backward References**: In the dataset, connections showing which previous messages are replied to

**Candidates**: Remaining nodes that haven't been assigned to a thread yet

**Top-K Selection**: Selecting the K messages with highest predicted probabilities

**Teacher Forcing**: Using ground truth to guide training (vs. letting model choose freely)

**Keep/Discard Head**: Output head that predicts which nodes to remove after each prediction

**F1 Score**: Harmonic mean of precision and recall (main evaluation metric)

**Burst**: Group of messages within 5 minutes (from chunking algorithm)

**Section**: Group of bursts within 30-45 minutes (from chunking algorithm)

**Micro-thread**: Individual conversation thread within a chunk (from chunking algorithm)

---

## Contact & Resources

**Dataset Source**: IRC Disentanglement Dataset
- Paper: "Disentangling Conversations" (Kummerfeld et al., 2019)
- Location: `dataset/irc-disentanglement/`

**Related Work**:
- Time-Gapped HAC baseline: `algorithm_docs/baseline.md`
- Discourse marker analysis: Needed (future work)
- Graph-based conversation modeling: Research in progress

**Evaluation Tools**:
- F1 Score: `model/loss_function.py:compute_loss_f1()`
- Dataset evaluation scripts: `dataset/irc-disentanglement/tools/evaluation/`

---

## Notes for Claude Code

This project is in **active development** with the following characteristics:

1. **Experimental Code**: The architecture is evolving based on training results
2. **Documentation-First**: Check `algorithm_docs/` before making major changes
3. **Metrics-Driven**: All changes should be evaluated against F1 score and baseline
4. **Iterative Design**: The approach has pivoted several times (see `algorithm_doc.md`)

**When helping with this project**:
- Prioritize model architecture and training improvements
- Compare all changes against the baseline (72-76% F1)
- Maintain debug output for first N training examples
- Save all metrics to `results.json` for tracking
- Document major architectural changes in `algorithm_docs/`

**Secondary priorities** (mention but don't focus on):
- RAG system improvements
- Chunking algorithm refinements
- UI development (Chainlit)

The goal is to build a conversation disentanglement model that **significantly outperforms** the rule-based baseline, reaching 80-85% F1 score through learned representations.
