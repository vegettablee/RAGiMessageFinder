from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


# main orchestrator class per contact 

@dataclass
class DataState:
  # ───────────────────────
  # Meta / Control
  # ───────────────────────
  session_id: str
  contact_phone: str # shows currently selected contact 
  contact_name: str
  # Readiness flags
  raw_loaded: bool = False
  normalized_ready: bool = False
  time_ready: bool = False
  threads_ready: bool = False
  rag_ready : bool = False

  # 1st layer, just has raw message data (tuples from database)
  raw_messages: list = field(default_factory=list)
  max_messages: int = 0

  # 2nd layer, normalizes the raw data 
  normalized_messages: List[NormalizedMessage] = field(default_factory=list)

  # 3rd layer, used for time analytics, do this later, for now focus on the 4th layer
  time_analytics: Optional[TimeAnalytics] = None

  faiss_index: Optional[Any] = None
  message_lookup: Optional[Dict[str, dict]] = None # used for chunk embeddings and retrieval
  faiss_dim: int = 384  # default dimension for multi-qa-MiniLM-L6-cos-v1
  threads: List[Thread] = field(default_factory=list)

  errors: List[str] = field(default_factory=list)
  

# base data classes 
@dataclass(frozen=True) 
class NormalizedMessage:
  message_id: str
  contact_id: str
  sender: str
  timestamp: datetime
  order_index: int
  content: str

@dataclass
class Thread:
  thread_id: str
  message_ids: List[str]
  start_time: datetime
  end_time: datetime
  metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VectorChunk:
  chunk_id: str
  thread_id: str
  text: str
  metadata: Dict[str, Any] = field(default_factory=dict)

# analytics classes 
@dataclass
class TimeAnalytics:
  activity_over_time: Dict[str, int] = field(default_factory=dict)
  response_times: Dict[str, float] = field(default_factory=dict)
  gaps: List[float] = field(default_factory=list)
  bursts: List[Dict[str, Any]] = field(default_factory=list)
