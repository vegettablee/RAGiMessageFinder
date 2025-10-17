# this file takes the messages from message_loader and puts them into txt file to embed, also returns all the text 

import message_loader
import os 
import sys 
from data.processing.format import formatMyMessage, formatSenderMessage
from chunking import chunk_messages

processing_path = os.getcwd() 
print(processing_path)

project_dir = "/Users/prestonrank/RAGMessages"
sub_dir = "/Users/prestonrank/RAGMessages/data/processing"

file_name = 'data.txt'
MESSAGE_NUM = 10

def addToTextFile(phone_number : str, messages_per_subject : int, subject_name : str): 
  file = open(file_name, 'w')
  data = message_loader.getMessagesBySubject(phone_number, messages_per_subject)
  chunks = chunk_messages(data)
  print("number of messages from data : " + str(len(data)))
  counter = 0
  blank_idx = 0
  for entry in range(len(data)): 
    if(data[entry][1] == 'me'):
      file.write(formatMyMessage(data[entry][3], data[entry][0]))
    else : 
      if entry == len(data) - 1: # last index
        if entry == messages_per_subject - 1:
           file.write(formatSenderMessage(subject_name, data[entry][3], data[entry][0]))
        else : 
           file.write(formatSenderMessage(subject_name, data[entry][3], data[entry][0]))
      else : 
        file.write(formatSenderMessage(subject_name, data[entry][3], data[entry][0]))

  if messages_per_subject > len(data): # data did not return all messages, fill rest with null to avoid indexing issues
     for index in range(messages_per_subject - len(data)): 
        file.write("[NULL]\n")
        counter += 1
   
  print(data[0])
  print("messages written to data txt : " + str(len(data)))
  print("[NULL] values added to text file " + str(counter))

def getTextFile(sentences_per_embedding: int):
    with open(file_name, 'r') as file:
        read = file.readlines()
    
    all_text = []
    small_batch = []
    
    for line in read:
        # Clean the line
        clean_line = line.rstrip('\n')
        small_batch.append(clean_line)
        
        # When batch is full, save it and start new batch
        if len(small_batch) == sentences_per_embedding:
            all_text.append(small_batch)
            small_batch = []
    
    # Add any remaining items
    if small_batch:
        all_text.append(small_batch)
    return all_text

def getTextFileLine(index : int, index_multiplier : int): 
   with open(file_name, 'r') as file:
      read = file.readlines() 
   sentences = []
   index = index * index_multiplier
   if index_multiplier > 1:
      for i in range(0, index_multiplier): # to account for 0 edge case when adding indexes
        sentences.append(read[index + i])
   else : 
      sentences.append(read[index])

   return sentences # file starts at line 0
      
# addToTextFile("9365539666", 10)
#text = getTextFile(2)
# for line in text:
# print(line)
# raw data is in the form of : 
# ('2025-07-21 13:33:48', '+19365539666', '', 'Nah it’s okay, I found another pan 🤞', '+19365539666')
