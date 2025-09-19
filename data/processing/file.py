# this file takes the messages from message_loader and puts them into txt file to embed, also returns all the text 

import message_loader
import os 
import sys 

processing_path = os.getcwd() 
print(processing_path)

project_dir = "/Users/prestonrank/RAGMessages"
sub_dir = "/Users/prestonrank/RAGMessages/data/processing"

file_name = 'data.txt'
MESSAGE_NUM = 10

def addToTextFile(phone_number : str, messages_per_subject : int): 
  file = open(file_name, 'w')
  data = message_loader.getMessagesBySubject(phone_number, messages_per_subject)
  for entry in data: 
    if(entry[1] == 'me'):
      file.write("[ME] " + str(entry[3] + "\n"))
    else : 
      file.write("[THEM] " + str(entry[3] + "\n"))
  print("messages written to data txt : " + str(len(data)))


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
      for i in range(1, index_multiplier + 1): # to account for 0 edge case when adding indexes
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





