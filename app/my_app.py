from rag_agent import RagAgent
import gradio as gr
import time
from utils.mongodb import get_collection
import uuid
from utils.logger import LOG

collection = get_collection()
rag_agent = RagAgent(collection)

session_id = None

def slow_echo(user_message, history):
    global session_id
    if session_id is None:
        session_id = str(uuid.uuid4())
        LOG.info(f"history is empty, create a new session_id:::::::: ,{session_id}")
    response = rag_agent.response_to_user(user_message, session_id)
    for i in range(len(response)):
        time.sleep(0.01)
        yield "" + response[: i + 1]


with gr.Blocks() as demo:
    gr.Markdown("# Airbnb Chatbot")
    chatbot = gr.Chatbot(
        type="messages",
        height=600,
        bubble_full_width=False,
        show_label=True,
        show_copy_button=True
    )

    gr.ChatInterface(   
        fn=slow_echo,
        chatbot=chatbot,
        type="messages",
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
        multimodal=True,
        textbox=gr.MultimodalTextbox(file_count="single", file_types=["image"], sources=["upload"]),
    )
    with gr.Row():
        clear_session = gr.Button("New Chat")
        
    def reset_session():
        global session_id
        session_id = None
        return None
        
    clear_session.click(
        fn=reset_session,
        outputs=chatbot
    )


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")  # 启动界面并设置为公共可访问
