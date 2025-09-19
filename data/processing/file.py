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
  file = open(file_name, 'r') 
  read = file.readlines()
  all_text = []
  counter = 0
  small_batch = []
  for index in range(len(read)):
    if counter < sentences_per_embedding: 
      if read[index][-1] == '\n':
        small_batch.append(read[index][:-1])
      else: 
        small_batch.append(read[index])
    else: 
        all_text.append(small_batch) 
        small_batch = [] # reset the small batch
        counter = 0
    counter += 1

  if len(small_batch) > 0: 
    all_text.append(small_batch) 

  return all_text
      
addToTextFile("9365539666", 10)
# getTextFile(2)
# raw data is in the form of : 
# ('2025-07-21 13:33:48', '+19365539666', '', 'Nah it’s okay, I found another pan 🤞', '+19365539666')




