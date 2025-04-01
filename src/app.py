from vector_search_mongodb import VectorSearchMongoDB
from multimodal_vector_search_mongodb import MultiModalVectorSearchMongoDB
from rag_agent_multimodal import RagAgent
import gradio as gr
from pymongo import MongoClient
import os
import time

uri = os.getenv('MONGODB_URI')
client = MongoClient(uri)

db_name = 'airbnb_dataset'
collection_name = 'airbnb_embeddings'

db = client[db_name]
collection = db[collection_name]

# vector_search = VectorSearchMongoDB(db, collection)
vector_search = MultiModalVectorSearchMongoDB(db, collection)
rag_agent = RagAgent(db, collection, vector_search)


def slow_echo(query, history):
    response = rag_agent.handle_user_query([query['text'],query['files'][0]])
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
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"], sources=["upload", "microphone"])
)


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")  # 启动界面并设置为公共可访问
