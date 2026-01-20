from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from backend.rag.pipeline.classes import DataState
import chainlit as cl

@dataclass
class Session_State:
  user_id: Optional[str] = None
  contacts: List[Dict[str, str]] = field(default_factory=list)
  selected_contacts: List[str] = field(default_factory=list)
  qa_chain: Any = None  # Required - will be initialized on chat start

def get_session_state() -> Session_State:
    state = cl.user_session.get("state")
    if state is None:
        state = Session_State(
            user_id=cl.user_session.get("user_id"),
            contacts=cl.user_session.get("contacts", []),
            selected_contacts=cl.user_session.get("selected_contacts", []),
            qa_chain=cl.user_session.get("qa_chain"),
        )
        cl.user_session.set("state", state)
    return state

@dataclass 
class Contact_Data_States: # contains all of the contact-specific message data/analytics 
   DATA_REGISTRY : map 
   # in the form of : 
   # contact_name : DataState # where DataState is a class tied to the contact info, using a hashmap for efficient routing 
   # add metadata class 

def get_data_state() -> Contact_Data_States: 
    dataState = cl.user_session.get("data_state")
    if dataState is None: 
       dataState = Contact_Data_States( 
          DATA_REGISTRY = {}
       )
       cl.user_session.set("data_state", dataState)
    return dataState



