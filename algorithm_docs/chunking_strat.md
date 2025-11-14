## Approaches for chunking algorithm 
Throughout this project, I'm experimenting with different chunking strategies to best keep relevant texts together for better retrieval.
1. Separating by bursts(messages within 5 minutes), and separating by sections(conversation ended, messages more than 45 minutes apart)

Specifically, two time thresholds for sections and bursts : MAX_SECION_TIME, LINE_BURST_TIME
For a conversation with three different conversations at different times, there should be : 
- 3 sections, x amount of bursts per section, y amount of messages per burst

- Compute time difference between current message and next message
- If within LINE_BURST_TIME(5 minutes), put them into the same burst 
- Else, check if they are between the MAX_SECTION_TIME(45 minutes) and LINE_BURST_TIME(5 minutes)
- - - If within boundary, update the number of bursts with the collected messages within LINE_BURST_TIME
- - - Else, not in boundary(conversation may have ended), compare similarity between current message and next message
- - - - if they are similar, stay on the same section 
- - - - else, move to a different section, clear the previous bursts

Tested with : 
text_similarity threshold = 0.15, 0.22
burst_similarity threshold = 0.4, 0.5, 0.35, 0.32

Issues regarding similarity comparisons : 
text_similarity = 0.15
- too permissive, over-merges texts 

burst_similarity threshold = 0.5, 0.4
- too high, conversational texts generally output lower scores due to slang and less content to work with.
- 0.35 is was closer, but similar texts tend to hover around 0.3, many do not reach 0.35

Best parameters :
text_similarity threshold = 0.22
similarity threshold = 0.35

Issues regarding long-term conversation thread topic preservation : 
- with the given parameters, the algorithm is great at marking topic shifts between bursts, but sections 
- do not contain threads that have the entire conversation for a given topic, bursts tend to only be 2-3 messages with each 
- burst just marking a different topic thread, rather than a burst having an entire relevant conversation thread 

However, this chunking algorithm is great at separating topics by bursts, just no long-term dependencies. 


2. In addition to partitioning into sections with respective bursts. Compare neighboring bursts with individual messages from each burst with each message from the adjacent burst, keep track of conversation micro-thread. Keep track of all micro-threads for section.

Tested with : 

text_similarity threshold = 0.22
thread_similarity threshold = 0.3

Example : 

time_idxs = [ 
  { "time_stamp", index1 },
  { "time_stamp", index2 },
]
- where each timestamp is an index corresponding to an index in micro-threads, only store the date, not hour/minute/seconds
- compare against the first and last element of each thread's time stamp(isolated micro-thread within boundary)

micro-threads = [
  [thread1], 
  [thread2],
]

Section = [ 
  [ # burst1
    "Hey, did you do the assignment yet?",
    "Also, are you going to the event later?" 
  ], 
  [ # burst2
    "I haven't started yet, I'll do it later.", 
    "It depends, let me check if I'm busy." 
  ] 
]

Compare every element in burst1 with every element in burst2
- if similar add both messages into micro-threads 
- - if similar with multiple messages from burst2, add those messages to the micro-thread as well 
- - TO DO : figure out how to merge multiple messages without having duplicate matches 
- - TO DO : add a time decay, so messages that are truly similar but further apart are more truthful 
- else, add the individual element from burst1 to micro-threads


After element is compared, find the micro-threads with length of 1, and find the date of the micro-thread. 
- find the closest indexes(set a max like 2) to the micro-thread, and compare each message of the isolated micro-thread with the longer micro-threads, compute average embedding score for comparing micro-threads
- whichever average score ends up being higher, reconstruct based on order 


With this approach, I am going to try a smaller decay rate that only applies after 15 minutes, because I ran log mean, standard deviation and a box plot to see the spread of the data, the variability, and percentiles. This is the results : 

STATISTICAL SUMMARY
Mean:              98.91 minutes (1.65 hours)
Std Deviation:     1104.55 minutes (18.41 hours)
Std/Mean Ratio:    11.17x

Total messages:    4690
Min time diff:     0.00 minutes
Max time diff:     39929.73 minutes (665.50 hours)

Percentiles:
  25th percentile: 0.13 minutes (0.00 hours)
  50th percentile: 0.33 minutes (0.01 hours)
  75th percentile: 12.67 minutes (0.21 hours)
  90th percentile: 91.61 minutes (1.53 hours)
  95th percentile: 294.84 minutes (4.91 hours)
  99th percentile: 1273.67 minutes (21.23 hours)

Using this, the cutoff between 75th and 90th percentile is by far the largest, so using this information, the decay needs to start after 15 minutes. The decay needs to be slow though, because messages tend to be very skewed, so in the cases where it might reach a day, the decay doesn't completely ruin the similarity computation. Using this, the decay formula is as follows : 


Remainder : Morning! Did you upload the final report draft yet?
Max score : 0.5926762223243713 at thread 0, message index 1
Complement : Yeah, I just uploaded the report this morning.

Remainder : Morning! Did you upload the final report draft yet?
Max score : 0.26362255215644836 at thread 1, message index 0
Complement : Also, are we still on for coffee later?

Remainder : I added the budget section to the spreadsheet last night.
Max score : 0.5645936131477356 at thread 0, message index 2
Complement : The budget section looks solid, maybe expand the notes under Q3.

Remainder : I added the budget section to the spreadsheet last night.
Max score : 0.1522873044013977 at thread 1, message index 0
Complement : Also, are we still on for coffee later?

## Pivoting Chunking Algorithm to individual micro thread extraction, keeping topics shifts but training a separate ML model to extract conversation threads from smaller bursts

I decided to pivot and use the IRC conversation disentanglement dataset. It would provide a significantly better and more quantifiable metric, while also making it a lot easier to create a separate ML model for disentangling conversations. 

For the actual ML model, I plan on using a reinforcement learning approach, where the reward is a ratio between the global sum of cosine similarity from all threads individually and individual thread scores. 

So, let's say the conversation is something like : 

Person : Hey, do you want to get coffee later? 
Person : Also, I need you to finish the report
Me : I'm good on coffee, I've been awake for so long already. 
Me : But yeah I finished the report 

Individual thread sum would be calculated as : 

1st thread : cosine similarity between both messages is 0.5
Person : Hey, do you want to get coffee later? 
Me : I'm good on coffee, I've been awake for so long already. 

2nd thread : cosine similarity between both messages is 0.6 
Person : Also, I need you to finish the report
Me : But yeah I finished the report 

Two separate sums, but using their individual scores as a reward/loss-function compared to how the model computes threads. 

Additionally, global sum of threads : 

1st thread : cosine similarity between both messages is 0.5
Person : Hey, do you want to get coffee later? 
Me : I'm good on coffee, I've been awake for so long already. 

2nd thread : cosine similarity between both messages is 0.6 
Person : Also, I need you to finish the report
Me : But yeah I finished the report 

Global sum would be 1st thread + 2nd thread score = 0.5 + 0.6 = 1.1 

So everytime the model makes a set of threads, compute their global sum relative to the ground truth from the IRC dataset. 

Parameters to think about : 

Ratio between global sum and individual thread sums, determines the reward and loss function.

Did some research, this will not work because the reward functions may be vague, and the cosine similarity is highly dependent on the root message. Because even if you use global sums, they are very sensitive, and the model will likely not be able to properly learn because the reward functions are just vague in general. 



Next possible approach : Teacher forcing, then transferring over to reinforcement learning for refinement.

Where messages are modeled as a decision tree. And the model picks an order of nodes, represented by messages.
Ground truth threads are sorted in order, and the model predicts these in order. 

Say for example, turn these into nodes : 
Node 1 [14:32] <BurgerMann> hey how do I install nvidia drivers?
Node 2 [14:33] <delire> BurgerMann: sudo apt-get install nvidia-drivers
Node 3 [14:33] <Seveas> anyone know if mysql is down?
Node 4 [14:34] <BurgerMann> delire: which version should I use?
Node 5 [14:34] <delire> Seveas: works for me
Node 6 [14:35] <techguy> BurgerMann: try nvidia-driver-470
Node 7 [14:35] <Seveas> delire: hmm weird, must be my connection
Node 8 [14:36] <BurgerMann> thanks both!

First two ground truth threads would be : 
Node 1 [14:32] <BurgerMann> hey how do I install nvidia drivers?
Node 2 [14:33] <delire> BurgerMann: sudo apt-get install nvidia-drivers

Node 3 [14:33] <Seveas> anyone know if mysql is down?
Node 5 [14:34] <delire> Seveas: works for me

In cases where one message may have two different connections. Use an associated counter, so whenever the model uses this node that connects to multiple other nodes, decrement the counter. This way, the model can understand that messages can directly relate to other messages. 

Model predicts nodes, then compare with ground truth threads like :

Prediction : [1, 3]
Ground Truth : [1, 2]
Compare accuracy ratio : 0.5

When accuracy ratio >= 0.75, do not teacher force and let the model predict the next set of nodes. 
When accuracy ratio < 0.75, teacher force the ground truth, then move on to predicting the next thread. 

Elaborating on the example, since accuracy ratio is < 0.75, feed the model the ground truth and remove used nodes.

Prediction : [1, 3]
Ground Truth : [1, 2] 

Feed model ground truth [1, 2] as the first chosen thread 
Remaining nodes become : [3, 4, 5, 6, 7, 8]

Predict next partial thread and iterate until no nodes remain. 

Potential features for prediction(only use when simple approach yields decent results): 
- Embedding of message
- Embeddings of existing threads (average of messages)
- Cosine similarity between message i and each thread
- Number of messages in each thread
- Time gap to most recent message in each thread
- Speaker patterns

Switch to hybrid learning approach when supervised learning starts yielding higher averages across batches, 75 percent accuracy for now. 
Specifically, once a given thread is predicted with high enough accuracy, switch to reinforcement learning instead of teacher-forcing/supervised for the rest of the messages/threads.

Now in this hybrid learning approach, using the previous example : 

First two ground truth threads : 

Node 1 [14:32] <BurgerMann> hey how do I install nvidia drivers?
Node 2 [14:33] <delire> BurgerMann: sudo apt-get install nvidia-drivers
Model predicts : [1, 2], accuracy = 100%

Switch to reinforcement learning for the remaining threads, so for the rest of the thread, there is no teacher forcing is the model predicts wrong or inaccurately. 

Node 3 [14:33] <Seveas> anyone know if mysql is down?
Node 5 [14:34] <delire> Seveas: works for me
Node 7 [14:35] <Seveas> delire: hmm weird, must be my connection

Model predicts : [3, 5, 8], accuracy = 70% 
Now, no teacher force happens, and the model moves on until all remaining nodes are used, regardless of accuracy. 

Neural network dimensions for first implementation : 

Input feature dim : 768(embeddings) 
Layer 1 : 256
Layer 2 : 128 
Layer 3 : 64
Output : num_threads + 1 

Explanation : 

Input feature dim is going to be the embedding of the remaining nodes(messages) in the conversation. 

Layers 1 - 3 are for learning implicit patterns about the conversational structure. 

Output : number of current threads and the option of creating a new thread  
- use a probability distribution over all possible choices, decode the logits from the hidden state of layer 3 
- specifically, a probability distribution over all remaining nodes, scores that have a score greater than 0.5 get added to the thread classification, this is going to be BCE, where the nodes use a sigmoid activation function 



In addition to having the model predict the correct thread, add output field where, after the model actually predicts the correct thread. associate each logit with a true and false field, where this handles if we should keep the given node for the next iteration. I'm doing this mainly to also have the model learn when to keep/discard nodes, and then incorporating this into the loss function. 

