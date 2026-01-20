# Message formatting utilities

def formatMyMessage(message_text: str, timestamp: str) -> str:
    """
    Format a message sent by me.
    
    Args:
        message_text: The message content
        timestamp: Message timestamp
    
    Returns:
        Formatted message string
    """
    return f"[{timestamp}] Me: {message_text}\n"


def formatSenderMessage(sender_name: str, message_text: str, timestamp: str) -> str:
    """
    Format a message sent by another person.
    
    Args:
        sender_name: Name of the sender
        message_text: The message content
        timestamp: Message timestamp
    
    Returns:
        Formatted message string
    """
    return f"[{timestamp}] {sender_name}: {message_text}\n"
