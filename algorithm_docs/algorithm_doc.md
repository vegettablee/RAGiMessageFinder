## Project Documentation 

This file is mainly responsible for documenting my journey with trying to solve the task of conversation disentanglement. All of these portions of the algorithm still need fine-tuning to be used with text-messages. For now, I'm using conversations with richer semantic content to make sure it can pick up. 

Bursts refer to groups that are within 5 minutes. 
Chunks refer to bursts that are within 30 minutes or semantically similar. 
Micro-threads refer to individual threads that exist within chunks. 

Computing the topic shifts : 

- I've finished this portion, the solution can be seen inside of topic_shift.py, at a high-level the algorithm does : 
- - Group messages within a threshold(5 minutes) to create bursts
- - Use these groups within a general threshold(30 minutes) to create bigger chunks 
- - Compare the last message to the next message that is over the general threshold(30 minutes) with cosine similarity
- - - If they are similar, merge them because even though they may be separated by time, they are still similar

Afterwards, chunks returned should be approximately separated by topic. 





