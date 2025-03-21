from vector_search_mongodb import VectorSearchMongoDB
from rag_agent import RagAgent
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

vector_search = VectorSearchMongoDB(db, collection)
rag_agent = RagAgent(db, collection, vector_search)


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
)


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")  # 启动界面并设置为公共可访问
