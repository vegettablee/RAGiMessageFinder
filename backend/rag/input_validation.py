import sys
import os
# Add parent directory to path for data imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.processing.message_loader import getMessagesBySubject

def handle_contact_input(subject_name, subject_phone) -> bool: 
  subject_name = subject_name.strip() 
  subject_phone = subject_phone.strip() 
  exists = validate_phone_number(subject_phone)
  return exists


def validate_phone_number(subject_phone) -> bool: 
  messages = getMessagesBySubject(subject_phone, 1)
  print(messages)
  if len(messages) == 0: 
    return False
  else: 
    return True 

