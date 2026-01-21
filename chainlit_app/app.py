import chainlit as cl

# Import subject_book directly (in same directory)
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.input_validation import handle_contact_input
from session_state import get_session_state, get_data_state
from contacts import update_settings
from backend.rag.rag import initialize_embedding_model, create_query_vector, initialize_rag_pipeline
from backend.rag.router import route_contact, multi_query_translation

# Placeholder functions - implement your RAG logic here
subject_phone = "9365539666"
subject_name = "Paris"

async def query_rag_pipeline(qa_chain, user_query, selected_contacts):
    try:
        # Extract components from qa_chain
        embedder = qa_chain['embedder']
        llm = qa_chain['llm']
        dimension = qa_chain['dimension']

        # Route query to selected contacts and retrieve relevant threads
        q = multi_query_translation(user_query)
        queries = q[1]
        category = q[0]
        all_results = []
        # Format retrieved threads as context for LLM
        context_text = ""
        source_documents = []


        for contact in selected_contacts:
            contact_phone = contact['phone']
            contact_name = contact['name']
            for query in queries:    
                doc_results = route_contact(selected_contacts, query)
                if contact_phone in doc_results:
                    retrieved_threads, message_lookup = doc_results[contact_phone]

                if retrieved_threads:
                    context_text += f"\n\n=== Conversations with {contact_name} ===\n"

                    for thread in retrieved_threads:
                        # Build thread conversation text
                        thread_text = f"\nConversation from {thread.start_time} to {thread.end_time}:\n\n"

                        # Get all messages in thread
                        for message_id in thread.message_ids:
                            msg_data = message_lookup.get(message_id)
                            if msg_data:
                                sender = msg_data['sender']
                                text = msg_data['text']
                                timestamp = msg_data['timestamp']
                                thread_text += f"[{timestamp}] {sender}: {text}\n"

                        context_text += thread_text

                        # Add to source documents for display
                        source_documents.append({
                            'contact_name': contact_name,
                            'content': thread_text
                        })

        # Build LLM prompt with context
        from langchain_core.messages import HumanMessage

        if context_text:
            prompt = f"""You are a helpful assistant that answers questions about iMessage conversations.

Below are relevant conversation threads retrieved from the user's messages:

{context_text}

Based on the above conversation history, please answer the following question:
{user_query}

Provide a clear, concise answer. Reference specific conversations by contact name and time when relevant."""
        else:
            prompt = f"""You are a helpful assistant that answers questions about iMessage conversations.

No relevant conversation threads were found for the query: {user_query}

Please let the user know that no relevant messages were found."""

        messages = [HumanMessage(content=prompt)]
        response_text = await llm.ainvoke(messages)

        return {
            'result': response_text.content,
            'source_documents': source_documents
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            'result': f'Error: {str(e)}\n\nDetails:\n{error_details}',
            'source_documents': []
        }

@cl.on_chat_start
async def start():
    # Check if this is first session initialization
    is_first_session = cl.user_session.get("state") is None

    # Get or create session state
    state = get_session_state()
    data_state = get_data_state()
    state.qa_chain = await initialize_rag_pipeline()

    cl.user_session.set("state", state)
    cl.user_session.set("data_state", data_state)

    # Use update_settings() so settings refresh dynamically when contacts are added
    await update_settings()

    # Only show welcome message on first session
    if is_first_session:
        # Create custom analytics button element
        analytics_button = cl.CustomElement(
            name="AnalyticsButton",
            props={},
            display="inline"
        )

        welcome_message = """**ThreadFinder** - iMessage query system for your iMessage conversations.

Add a contact via Settings to get started. Use `/analytics` for conversation insights.
        """
        await cl.Message(content=welcome_message).send()
        # change this to be shown on command 


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages from the user."""
    # Get session state
    state = get_session_state()

    # Check which contacts are selected
    if not state.selected_contacts:
        await cl.Message(
            content="⚠️ **No contacts selected.**\n\nPlease go to ⚙️ Settings and check the box next to at least one contact to query."
        ).send()
        return

    # Build list of selected contact dicts (not just phone numbers)
    selected_contact_dicts = [c for c in state.contacts if c['phone'] in state.selected_contacts]
    selected_names = [c['name'] for c in selected_contact_dicts]

    # Create empty message to update
    msg = cl.Message(content="")
    await msg.send()

    # Query the RAG pipeline with selected contacts
    response = await query_rag_pipeline(state.qa_chain, message.content, selected_contact_dicts)

    # Update message with response and show which contacts were queried
    queried_contacts = ", ".join(selected_names)
    msg.content = f"**Querying:** {queried_contacts}\n\n{response['result']}"
    await msg.update()

    # Display source documents if available
    if response.get("source_documents"):
        print(f"DEBUG: Found {len(response['source_documents'])} source documents")

        source_elements = []
        for i, doc in enumerate(response["source_documents"]):
            contact_name = doc.get('contact_name', 'Unknown')
            content = doc.get('content', '')

            print(f"DEBUG: Source {i+1} - Contact: {contact_name}, Content length: {len(content)}")

            if content:  # Only add if content exists
                source_elements.append(
                    cl.Text(
                        name=f"📱 {contact_name} - Thread {i+1}",
                        content=content,
                        display="inline"  # Changed from "side" to "inline"
                    )
                )

        if source_elements:
            await cl.Message(
                content=f"📋 **Source Conversations:** ({len(source_elements)} threads found)",
                elements=source_elements
            ).send()
        else:
            print("DEBUG: No source elements to display (all content was empty)")
            await cl.Message(content="📋 No source conversations found with content.").send()