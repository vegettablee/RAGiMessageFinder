import chainlit as cl
from session_state import get_data_state

async def show_analytics_selector(selected_contacts: list):
    """
    Display analytics selection UI with checkboxes and confirm button.

    Args:
        selected_contacts: List of selected contact dicts
    """
    # Initialize analytics selections in session if not exists
    if cl.user_session.get("analytics_selections") is None:
        cl.user_session.set("analytics_selections", set())

    # Get contact names for display
    contact_names = [c['name'] for c in selected_contacts]
    contacts_display = ", ".join(contact_names) if contact_names else "No contacts selected"

    # Define available analytics options
    analytics_options = [
        {"id": "message_frequency", "label": "Message Frequency Over Time"},
        {"id": "response_time", "label": "Response Time Analysis"},
        {"id": "sentiment_analysis", "label": "Sentiment Analysis"},
        {"id": "topic_distribution", "label": "Topic Distribution"},
        {"id": "conversation_length", "label": "Conversation Length Statistics"},
        {"id": "word_cloud", "label": "Word Cloud"}
    ]

    # Build message content
    message_content = f"""## Analytics Dashboard

**Selected Contacts:** {contacts_display}

Choose analytics to run by clicking the options below, then confirm:
"""

    # Create action buttons for each analytics type + confirm
    actions = []
    for option in analytics_options:
        actions.append(
            cl.Action(
                name="toggle_analytics",
                value=option['id'],
                label=option['label'],
                payload={"analytics_id": option['id']}
            )
        )

    # Add confirm button
    actions.append(
        cl.Action(
            name="confirm_analytics",
            value="confirm",
            label="CONFIRM & RUN ANALYTICS",
            payload={"action": "confirm"}
        )
    )

    # Send message with action buttons
    await cl.Message(content=message_content, actions=actions).send()

@cl.action_callback("toggle_analytics")
async def on_toggle_analytics(action: cl.Action):
    """Toggle analytics selection"""
    selections = cl.user_session.get("analytics_selections")
    analytics_id = action.value

    if analytics_id in selections:
        selections.remove(analytics_id)
        await cl.Message(content=f"Deselected: {action.label}").send()
    else:
        selections.add(analytics_id)
        await cl.Message(content=f"Selected: {action.label}").send()

    cl.user_session.set("analytics_selections", selections)

@cl.action_callback("confirm_analytics")
async def on_confirm_analytics(action: cl.Action):
    """Handle analytics confirmation - receives all selected options"""
    selections = cl.user_session.get("analytics_selections", set())

    if not selections:
        await cl.Message(content="No analytics selected. Please select at least one option.").send()
        return

    # Convert set to list for processing
    selected_analytics = list(selections)

    await cl.Message(
        content=f"Running {len(selected_analytics)} analytics:\n" +
                "\n".join([f"- {a}" for a in selected_analytics])
    ).send()

    # TODO: Call analytics execution here with selected_analytics list
    # handle_analytics_execution(selected_analytics, contacts)

    # Clear selections after confirmation
    cl.user_session.set("analytics_selections", set())

def handle_command(command : str, selected_contacts : any) -> bool:
    if command.startswith("/analytics"):
        return True
    return False 

def handle_analytics(contact_name : str) -> tuple: 
  data_states = get_data_state() 
  if contact_name not in data_states.DATA_REGISTRY[contact_name]: # return if not found 
    return 

  data_state = data_states.DATA_REGISTRY[contact_name] 

  return () 
def is_command(query : str) -> bool: 
  if query.startswith("/"): 
    return True 
  else: 
    return False 