# this will hold the main algorithm for chunking the data into relevant chunks : 
# first, the data needs to be divided based on time intervals
# >30 minutes, create a section(as conversation may have died), create all sections
# <5 minutes, inside of each section check for multi-line bursts from the same sender, and create a sub-section for each section
# once conversation and multi-line bursts are sectioned 
# start from the first conversation section
# from the first section inside of the conversation section, check embedding similarity for 1 chunk above
# if similar enough, merge the chunks together
# continue to the next section if the current section is already a merged section, check once more for relevant chunks
# move up until all chunks have been compared and properly chunked
