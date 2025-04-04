from rag_agent import RagAgent
import gradio as gr
import time
from utils.mongodb import get_collection


collection = get_collection()
rag_agent = RagAgent(collection)

# TODO: 
# add restriction for search only by image

def slow_echo(query, history):   
    response = rag_agent.handle_user_query(query)
    for i in range(len(response)):
        time.sleep(0.01)
        yield "" + response[: i + 1]

demo = gr.ChatInterface(
    slow_echo,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
    # examples=[
    #     {"text": "No files", "files": []}
    # ],
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="single", file_types=["image"], sources=["upload"])
)


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")  # 启动界面并设置为公共可访问
