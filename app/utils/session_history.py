from langchain_core.chat_history import (
    BaseChatMessageHistory,  
    InMemoryChatMessageHistory,  
)


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    get the session history
    Args:
        session_id (str): the unique identifier of the session
    
    Returns:
        BaseChatMessageHistory: the chat history of the session
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]