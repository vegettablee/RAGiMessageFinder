# Main App Architecture 

Data flow : 

1. User opens the application, greeted with a prompt that allows them to enter a phone-number of the person they want to view. 
- Hit the database first, if nothing comes up, ask for another phone number. for now, don't add error handling 
- - If successful, ask for a name tied to the number, and continue with the rest of the application 
- - Else, ask for a different phone number and reprompt 

2. Phone number found, find the max amount of messages that a user actually has with that person, and ask for a range of how many messages they want to extract. For now, just take all of the messages that exist. 

3. Create a FAISS index based on the phone number. 

4. Start cleaning and formatting the messages into relevant, bigger chunks based on time gaps, cosine similarity, and other factors. 

5. Put all chunks into a hashmap, where each key is the index relative to the FAISS database.

6. Transform each chunk into a vector embedding in order based on the indexes, once all chunks have been completed, add them to the FAISS database. 

7. Once completed, show a clear indication, and connect the OpenAI API. 

8. Allow the user to query, forcing them to check the subjects they want to ask about. So if they wanted to ask a question about one other person, they would have that box checked off, multiple would do the same. 

9. Before each prompt, use the subjects checked as a meta-data index matcher for retrieval. 

10. Print LLM response along with data used to help make the answer. So even if the LLM is wrong, provide the original messages to see if it was accurate or not, or a reminder. 
