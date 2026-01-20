from backend.data.processing.message_loader import getMessagesBySubject
from backend.rag.pipeline.classes import DataState
from backend.rag.pipeline.pipeline import run_main_pipeline
from chainlit_app.session_state import get_data_state
import chainlit as cl
import faiss

# runs when a new contact is added, calls the entire data processing pipeline apart from calculating analytics 
def create_contact_db(contact_tuple : tuple, dim : int, MAX_MESSAGES : int) -> int: # returns number of messages added to the FAISS database 
  contact_name = contact_tuple[0]
  contact_phone = contact_tuple[1]
  contact_info = contact_tuple[2]

  # Generate a unique session_id for this contact
  import uuid
  session_id = f"{contact_name}_{contact_phone}_{uuid.uuid4().hex[:8]}"

  dataState = DataState(
      session_id=session_id,
      contact_name=contact_name,
      contact_phone=contact_phone,
      max_messages=1000,
      faiss_dim=dim
  )
  dataState = run_main_pipeline(dataState)

  all_data_states = get_data_state()
  
  all_data_states.DATA_REGISTRY[contact_name] = dataState # add to registry 

  cl.user_session.set("data_state", all_data_states) # save it to the data session 