## Approaches for chunking algorithm 
Throughout this project, I experimented with different chunking strategies to best keep relevant texts together for better retrieval.
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


Issues : 

For some reason, the chunking algorithm is good at finding the best match, but the algorithm is too greedy. To elaborate, once a message is claimed from the algorithm, there's no way for that message to be reconsidered even if the best match is not yet found. So I'm going to add a separate algorithm that takes two bursts, and tries to find the greatest thread similarity. Since similarity is done sequentially, depending on the order, the message, it can vary wildly what can be the threads that yield the highest similarity score. I can still use the original algorithm for best matches and more approximate micro-threads. 

Next Step : Create an algorithm that can find micro-threads such that the max global sum of all similarity scores is reached, and such that all micro-threads have been computed with their highest possible similarity score for individual threads.

I think I could still use the old algorithm for computing one to one matches with all of burst1 to burst2. Additionally, I could try to apply it with bigger threads and test out approximate micro-threading. Then, use the other algorithm on smaller bursts, then compute the relevant micro-threads. Starting off more general, finding topic shifts, then taking apart individual bursts would be ideal.

