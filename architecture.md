# RAGMessages - Message Retrieval and Analysis System

A Retrieval-Augmented Generation (RAG) system for analyzing and querying iMessage conversations using semantic search and intelligent chunking.

## Architecture Overview

This project implements a RAG pipeline that:
1. Extracts messages from macOS iMessage database
2. Chunks messages into semantically meaningful conversation sections
3. Embeds chunks using sentence transformers
4. Stores embeddings in FAISS for efficient similarity search
5. Provides a Chainlit UI for conversational querying

## System Flow

```
iMessage DB (chat.db)
    ↓
message_loader.py → Load & normalize messages
    ↓
chunking.py → Intelligent time-based & semantic chunking
    ↓
format.py → Format messages with timestamps & slang expansion
    ↓
rag.py → Create embeddings & FAISS index
    ↓
app.py → Chainlit UI for querying
```

---

## Core Components

### 1. Data Extraction Layer

#### `data/processing/message_loader.py`
**Purpose**: Extract and normalize messages from the macOS iMessage SQLite database

**Key Functions**:
- `getMessagesBySubject(subject, numberOfMessages)` - Retrieves messages for a specific contact
  - Handles phone number normalization (E.164 format)
  - Decodes `attributedBody` blobs for outgoing messages
  - Filters out attachments and reactions
  - Returns tuples: `(timestamp, sender, '', message_text, phone)`

**Features**:
- Phone number normalization (handles +1, 10-digit, 11-digit formats)
- AttributedBody blob parsing for rich text extraction
- Excludes system messages, reactions, and attachments
- Flexible contact matching (name, phone, chat identifier)

**Data Format Output**:
```python
('2024-10-08 01:31:13', 'me', '', 'Hey, how are you?', '+19365539666')
```

---

### 2. Message Chunking Layer

#### `data/processing/chunking.py`
**Purpose**: Intelligently segment conversations into semantically meaningful chunks using time-based and embedding similarity

**Algorithm**:
1. **Time-based Burst Detection** (`chunk_messages`)
   - `< 5 minutes`: Same burst (rapid-fire messages)
   - `5-30 minutes`: New burst within same conversation section
   - `> 30 minutes`: New conversation section (unless semantically similar)

2. **Semantic Similarity Checks**
   - `compute_text_similarity()`: Cross-section similarity (keeps related topics together across time gaps)
   - `compute_burst_similarity()`: Adjacent burst comparison (marks topic shifts)

3. **Microthread Detection** (`compute_microthreads`) - *In Progress*
   - Identifies sub-topics within conversation sections
   - Tracks which messages relate to which threads

**Constants**:
```python
MAX_SECTION_TIME = 30  # minutes
LINE_BURST_TIME = 5    # minutes
BURST_SIMILARITY_THRESHOLD = 0.32
TEXT_SIMILARITY_THRESHOLD = 0.22
```

**Output Structure**:
```python
validated_chunks = [
  [  # Section 1
    [burst1_msg1, burst1_msg2],  # Burst 1
    [burst2_msg1, burst2_msg2],  # Burst 2 (separate topic)
    ([burst3_msg1], [burst4_msg1])  # Bursts 3 & 4 (merged - similar topic)
  ],
  [  # Section 2
    [burst1_msg1, burst1_msg2]
  ]
]
```

#### `data/processing/display.py`
**Purpose**: Display chunking metrics and validated chunks for debugging

**Key Functions**:
- `display_conversation_sections(all_chunks)` - Shows initial time-based sections
- `display_validated_chunks(validated_chunks)` - Shows final chunks after similarity processing
- `format_message(msg_tuple)` - Formats individual messages for readable output

---

### 3. Message Formatting Layer

#### `data/processing/format.py`
**Purpose**: Format messages with human-readable timestamps and expand slang

**Key Functions**:
- `format_message(subject_name, message, timestamp)` - Converts raw timestamps to readable format
  - Input: `"2024-10-08 01:31:13"`
  - Output: `"Sent from Me on Tuesday, October 08 2024, 1:31 AM : message"`

- `removeSlang(message)` - Expands text slang using dictionary lookup
  - Example: "omw" → "on my way"

- `formatMyMessage(message, timestamp)` - Format outgoing messages
- `formatSenderMessage(subject_name, message, timestamp)` - Format incoming messages

**Dependencies**:
- `slang_dict.py`: Dictionary mapping slang to expanded forms
- `time_dict.py`: Month/day name mappings

#### `data/processing/file.py`
**Purpose**: Bridge between message loader and text file storage

**Key Functions**:
- `addToTextFile(phone_number, messages_per_subject, subject_name)`
  - Loads messages via `message_loader`
  - Formats messages using `format.py`
  - Writes to `data.txt`
  - Pads with `[NULL]` for consistent indexing

- `getTextFile(sentences_per_embedding)` - Reads text file and batches messages
- `getTextFileLine(index, index_multiplier)` - Retrieves specific message by index

---

### 4. RAG Pipeline Layer

#### `rag.py`
**Purpose**: Core RAG functionality - embeddings, FAISS indexing, and retrieval

**Key Functions**:

1. **`initialize_embedding_model(model_name)`**
   - Default: `"multi-qa-MiniLM-L6-cos-v1"` (384-dim)
   - Uses **Asymmetric Semantic Search** approach
   - Query: Short questions → Document: Longer context
   - Returns: `(embedder, dimension)`

2. **`load_data(subject_phone, subject_name, messages_per_subject, sentences_per_embedding)`**
   - Loads messages from file system
   - Batches sentences into chunks
   - Returns: `(corpus, index_multiplier)`

3. **`create_faiss_index(embedder, corpus, dimension)`**
   - Encodes corpus using `encode_corpus()`
   - Creates FAISS IndexFlatL2 (Euclidean distance)
   - Adds embeddings to index
   - Returns: `(index, embeddings)`

4. **`create_query_vector(embedder, query_text)`**
   - Encodes query using `encode_query()` (asymmetric)
   - Returns: Tensor of shape `(1, dimension)`

**Embedding Approach**:
- **Asymmetric Search**: Queries and documents encoded differently
- **Model**: MS MARCO pre-trained (multi-qa-mpnet-base-cos-v1)
- **Similarity**: Cosine similarity via normalized embeddings
- **Storage**: FAISS IndexFlatL2 for exact search

---

### 5. Application Layer

#### `app.py`
**Purpose**: Chainlit-based conversational UI for querying the RAG system

**Chainlit Event Handlers**:

1. **`@cl.on_chat_start` - `start()`**
   - Initializes RAG pipeline on session start
   - Calls `initialize_rag_pipeline()`:
     - Loads embedding model
     - Loads message data
     - Creates FAISS index
   - Stores `qa_chain` in user session

2. **`@cl.on_message` - `main(message)`**
   - Receives user query
   - Retrieves QA chain from session
   - Calls `query_rag_pipeline(qa_chain, query)`
   - Returns results with source documents
   - Displays sources in sidebar

**Configuration**:
```python
subject_phone = "9365539666"
subject_name = "Paris"
```

**TODO Items**:
- Complete RAG chain implementation
- Add LLM integration (OpenAI/Anthropic)
- Implement actual retrieval logic
- Add context window management

---

### 6. Experimental Notebook

#### `main.ipynb`
**Purpose**: Jupyter notebook for testing and experimentation

**Sections**:
1. **Embedding Setup** - Initialize BERT models for semantic search
2. **Data Loading** - Load messages via `file.py`
3. **FAISS Indexing** - Create and test vector search
4. **Query Testing** - Test retrieval with sample queries
5. **Similarity Analysis** - Compare embedding similarities

**Example Queries Tested**:
```python
"What time is the kickback?"
"all kickbacks around fall"
```

**Metrics**:
- Cosine similarity scores
- Euclidean distance scores
- Top-k retrieval results

---

## Data Flow Example

### End-to-End Message Processing

```python
# 1. Extract from iMessage DB
messages = getMessagesBySubject("9365539666", 5000)
# → [('2024-10-08 01:31:13', 'me', '', 'Hey!', '+19365539666'), ...]

# 2. Chunk messages intelligently
chunks, metrics = chunk_messages(messages, embedder)
# → [[burst1, burst2], [burst3]]

# 3. Format for embedding
formatted = formatMyMessage("Hey!", "2024-10-08 01:31:13")
# → "Sent from Me on Tuesday, October 08 2024, 1:31 AM : Hey!"

# 4. Create embeddings
corpus = [formatted_chunk for chunk in chunks]
index, embeddings = create_faiss_index(embedder, corpus, 384)

# 5. Query
query_vec = create_query_vector(embedder, "When did we talk about movies?")
D, I = index.search(query_vec, k=5)  # Returns top 5 similar chunks
```

---

## Project Structure

```
RAGMessages/
├── app.py                      # Chainlit UI application
├── rag.py                      # RAG pipeline core logic
├── main.ipynb                  # Experimentation notebook
├── data/
│   ├── raw/
│   │   └── chat.db            # macOS iMessage database
│   └── processing/
│       ├── message_loader.py  # DB extraction
│       ├── chunking.py        # Intelligent chunking
│       ├── display.py         # Metrics display
│       ├── format.py          # Message formatting
│       ├── file.py            # File I/O bridge
│       ├── test_data.py       # Test fixtures
│       ├── slang_dict.py      # Slang expansion
│       └── time_dict.py       # Time formatting
└── data.txt                    # Processed messages (temp)
```

---

## Key Design Decisions

### 1. Why Asymmetric Semantic Search?
Queries are typically short questions ("When did we argue?") while messages are longer contextual blocks. Asymmetric models optimize for this mismatch.

### 2. Why Multi-Level Chunking?
- **Time-based**: Preserves conversation boundaries
- **Burst detection**: Captures rapid exchanges
- **Semantic similarity**: Keeps related topics together across time gaps
- **Microthreads**: Identifies interleaved sub-conversations

### 3. Why FAISS IndexFlatL2?
- Exact search (no approximation)
- Fast for datasets < 1M vectors
- Simple L2 distance metric
- Can upgrade to IVF/HNSW for scale

### 4. Why Chainlit?
- Built for conversational AI
- Easy source document display
- Session management
- Streaming support

---

## Configuration

### Chunking Parameters
Adjust in `chunking.py`:
```python
MAX_SECTION_TIME = 30       # Conversation break threshold (minutes)
LINE_BURST_TIME = 5         # Burst detection threshold (minutes)
BURST_SIMILARITY_THRESHOLD = 0.32  # Burst merging threshold
TEXT_SIMILARITY_THRESHOLD = 0.22   # Cross-section similarity
```

### Embedding Model
Change in `rag.py`:
```python
model_name = "multi-qa-MiniLM-L6-cos-v1"  # 384-dim, fast
# Alternative: "multi-qa-mpnet-base-cos-v1"  # 768-dim, more accurate
```

### Data Loading
Configure in `app.py` or `main.ipynb`:
```python
subject_phone = "9365539666"
messages_per_subject = 5000
sentences_per_embedding = 6  # Messages grouped per chunk
```

---

## Current Status & TODOs

### ✅ Completed
- iMessage database extraction
- Time-based conversation chunking
- Burst detection with semantic similarity
- FAISS indexing and basic retrieval
- Chainlit UI scaffold
- Message formatting with slang expansion

### 🚧 In Progress
- `compute_microthreads()` - Interleaved topic detection
- Bug fix: Duplicate burst handling (recently fixed)

### 📋 Pending
- LLM integration for response generation
- Context window management
- Query rewriting for better retrieval
- Metadata filtering (date ranges, sentiment)
- Multi-contact support
- Conversation summarization
- Export/backup functionality

---

## Development Notes

### Testing Chunking
Use `test_data.py` for sample conversations:
```python
from test_data import message_chunks
from chunking import chunk_messages

for chunk in message_chunks:
    validated, metrics = chunk_messages(chunk, embedder)
```

### Viewing Results
The `display.py` module automatically prints:
- Conversation sections with burst counts
- Validated chunks structure (merged vs separate)
- Message formatting with timestamps

### Common Patterns

**Adding a new similarity metric**:
1. Update `compute_text_similarity()` or `compute_burst_similarity()`
2. Adjust threshold constants
3. Test with `test_data.py`

**Changing embedding model**:
1. Update `initialize_embedding_model()` in `rag.py`
2. Update dimension parameter
3. Regenerate FAISS index

**Adding new message sources**:
1. Create loader in `message_loader.py`
2. Update tuple format to match: `(timestamp, sender, '', text, contact_id)`
3. Test with `file.py` bridge

---

## Dependencies

### Core Libraries
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector similarity search
- `torch` - Tensor operations
- `chainlit` - Conversational UI
- `sqlite3` - iMessage DB access

### Utilities
- `pandas` - Date/time manipulation
- `numpy` - Numerical operations
- `langchain-community` - RAG utilities (optional)

---

## Running the Application

### Start Chainlit UI
```bash
chainlit run app.py
```

### Run Jupyter Notebook
```bash
jupyter notebook main.ipynb
```

### Test Chunking Algorithm
```bash
python data/processing/chunking.py
```

---

## Performance Considerations

- **Messages**: ~5000 per contact (configurable)
- **Embedding time**: ~1-2 seconds per 1000 messages (MiniLM-L6)
- **Index search**: <10ms for 10k vectors (IndexFlatL2)
- **Memory**: ~2MB per 1000 messages embedded (384-dim)

---

## Future Enhancements

1. **Advanced Retrieval**
   - Hybrid search (BM25 + semantic)
   - Re-ranking with cross-encoders
   - Temporal filtering

2. **Better Chunking**
   - Conversation topic modeling
   - Entity-based grouping
   - Sentiment-aware boundaries

3. **UI Improvements**
   - Timeline visualization
   - Multi-contact queries
   - Export conversations

4. **Analytics**
   - Conversation statistics
   - Topic trends over time
   - Relationship insights

---

## License & Contact

*Project by Preston Rank*
