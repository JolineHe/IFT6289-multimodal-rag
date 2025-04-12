from rag_agent import RagAgent
import gradio as gr
import time
from utils.mongodb import get_collection
import uuid
from utils.logger import LOG
from langchain.chat_models import ChatOpenAI
from utils.const_db_fields import PROPERTY_TYPES

collection = get_collection()
rag_agent = RagAgent(collection)

# TODO: 
# add restriction for search only by image
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
    chatbot = gr.Chatbot(type="messages")

    with gr.Row():
        rank_slider = gr.Slider(0.0, 1.0, step=0.1, label="review_rank",
                                       value=None) # review_scores.review_scores_rating
        search_dropdown = gr.Dropdown(
            choices=PROPERTY_TYPES,
            value=None,
            label="app_type"
    )
    gr.ChatInterface(   
        fn=slow_echo,
        chatbot=chatbot,
        type="messages",
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
        multimodal=True,
        textbox=gr.MultimodalTextbox(file_count="single", file_types=["image"], sources=["upload"]),
        additional_inputs=[rank_slider, search_dropdown],
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
