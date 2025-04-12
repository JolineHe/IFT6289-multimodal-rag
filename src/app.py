from rag_agent import RagAgent
import gradio as gr
import time
from utils.mongodb import get_collection
from utils.const_db_fields import PROPERTY_TYPES

collection = get_collection()
rag_agent = RagAgent(collection)

def slow_echo(query,history,review_rank,app_type):
    other_params = dict() if review_rank==0 and app_type is None else {'review_rank': review_rank,"property_type": app_type}
    response = rag_agent.handle_user_query(
        query,
        other_params = other_params
    )
    for i in range(len(response)):
        time.sleep(0.01)
        yield "" + response[: i + 1]

# demo = gr.ChatInterface(
#     slow_echo,
#     type="messages",
#     flagging_mode="manual",
#     flagging_options=["Like", "Spam", "Inappropriate", "Other"],
#     save_history=True,
#     # examples=[
#     #     {"text": "No files", "files": []}
#     # ],
#     multimodal=True,
#     textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"], sources=["upload", "microphone"]),
#     additional_inputs=[
#         gr.CheckboxGroup(choices=["Option A", "Option B", "Option C"], label="search type")
#     ]
# )

with gr.Blocks() as demo:
    # gr.Markdown("## Chatbot with Parameter Controls")

    with gr.Row():
        rank_slider = gr.Slider(0.0, 1.0, step=0.1, label="review_rank",
                                       value=None) # review_scores.review_scores_rating
        search_dropdown = gr.Dropdown(
            choices=PROPERTY_TYPES,
            value=None,
            label="app_type"
        )

    chatbot = gr.ChatInterface(
        fn=slow_echo,
        type="messages",
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
        save_history=True,
        textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"], sources=["upload", "microphone"]),
        additional_inputs=[rank_slider, search_dropdown],
        # title="Multimodal Search Bot",
        # description="Adjust parameters before sending a message."
    )


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")  # 启动界面并设置为公共可访问
