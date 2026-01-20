import chainlit as cl
import sys
import os
from session_state import get_session_state

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag.input_validation import handle_contact_input
from backend.rag.rag_data_utils import find_max_messages
from backend.rag.add_contact import create_contact_db


async def update_settings():
    """Update chat settings with current contacts and input fields."""
    # Get session state
    state = get_session_state()
    contacts = state.contacts
    selected_contacts = state.selected_contacts

    settings = []

    # Add contact input fields at the top
    settings.extend([
        cl.input_widget.TextInput(
            id="contact_name",
            label="➕ Add New Contact - Name",
            initial="",
            placeholder="Enter contact's name..."
        ),
        cl.input_widget.TextInput(
            id="contact_phone",
            label="➕ Add New Contact - Phone Number",
            initial="",
            placeholder="e.g., +19365539666 or 9365539666"
        ),
        cl.input_widget.TextInput(
            id="contact_info",
            label="➕ Add New Contact - Additional Info (Optional)",
            initial="",
            placeholder="e.g., College friend, coworker, etc."
        ),
    ])

    # Add contact selection section if there are contacts
    if contacts:
        # Add contact selection switches with cleaner formatting
        for idx, contact in enumerate(contacts):
            contact_id = f"select_{contact['phone']}"

            # Format: Name (phone) with optional info
            # First contact gets a prefix label
            if idx == 0:
                display_name = f"🔍 Select Contacts to Query | {contact['name']} ({contact['phone']})"
            else:
                display_name = f"{contact['name']} ({contact['phone']})"

            if contact.get('info'):
                display_name += f" - {contact['info']}"

            settings.append(
                cl.input_widget.Switch(
                    id=contact_id,
                    label=display_name,
                    initial=contact['phone'] in selected_contacts
                )
            )

    await cl.ChatSettings(settings).send()
  
@cl.on_settings_update
async def on_settings_update(settings):
    # handle adding new contacts + selecting contacts for queries 
    contact_name = (settings.get("contact_name") or "").strip()
    contact_phone = (settings.get("contact_phone") or "").strip()
    contact_info = (settings.get("contact_info") or "").strip()

    # Get session state
    state = get_session_state()

    # Validation guard rails  
    errors = []
    invalid_field = False
    is_blank = False

    if not contact_name:
        errors.append("⚠️ **Validation Error:** Contact name cannot be blank.")

    if not contact_phone.isdigit() or len(contact_phone) != 10:
        errors.append("⚠️ **Validation Error:** Phone number must be exactly 10 digits with no letters or special characters.")

    if not contact_name and not contact_phone and not contact_info: 
        is_blank = True

    if errors and is_blank is False: # throw errors if invalid fields + non-blank input 
        await cl.Message(content="\n\n".join(errors)).send()
        return
    else: 
        exists_in_db = handle_contact_input(contact_name, contact_phone)
        if exists_in_db is True: # continue with the rest of the code 
            pass 
        else: 
            await cl.Message(content=f"Database could not find any messages with {contact_phone}, contact not added.\n").send()
            return # 

    if is_blank is True:
        pass
    else:
        # Add new contact to list
        new_contact = {
            "name": contact_name,
            "phone": contact_phone,
            "info": contact_info
        }
        state.contacts.append(new_contact)

        # Auto-select the new contact
        state.selected_contacts.append(contact_phone)
        # Save updated state back to session
        cl.user_session.set("state", state)
        # Show confirmation
        await cl.Message(
            content=f"""✅ **Contact Added!**

**Name:** {contact_name}
**Phone:** {contact_phone}
{f'**Info:** {contact_info}' if contact_info else ''}

Contact is now selected for queries. Use the checkboxes in settings to toggle contact selection.
                """
        ).send()
        
        await cl.Message(content=f"Processing messages with contact {contact_name}...").send()
        num_messages = find_max_messages(contact_phone)
        await cl.Message(content=f"Found {num_messages}... Stand by until all messages have been organized.").send()
        # Rebuild settings with cleared inputs
        await update_settings()
        contact_tuple = (contact_name, contact_phone, contact_info)
        # use the contact info as context before asking the LLM for advice  
        faiss_index = create_contact_db(contact_tuple, state.qa_chain["dimension"], num_messages)
        # Return early after adding contact - don't process switches
        return

    # Update selected contacts based on switch states
    new_selected = []
    original_selected = state.selected_contacts.copy()
    for contact in state.contacts:
        switch_id = f"select_{contact['phone']}"
        if settings.get(switch_id, False):
            new_selected.append(contact['phone'])
    if set(original_selected) != set(new_selected) : # if the user selected a new query, tell them what they selected 
        formatted = ", ".join(new_selected)
        await cl.Message(content=f"{formatted} is now selected for queries. Use the checkboxes in settings to toggle contact selection.").send()
    state.selected_contacts = new_selected

    # Save updated state back to session
    cl.user_session.set("state", state)
