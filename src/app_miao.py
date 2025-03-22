import gradio as gr
from PIL import Image

class Server(object):
    def __init__(self):
        self.collection = None
        self.db = None

    def handle_text(self, text):
        pass


def process_inputs(image, text1, text2, text3):
    """Processes the image and multiple text inputs."""
    img = Image.open(image)
    # Combine user inputs
    user_texts = f"Text 1: {text1}\nText 2: {text2}\nText 3: {text3}"

    return img, user_texts, text1


if __name__ == "__main__":
    # Create Gradio interface
    iface = gr.Interface(
        fn=process_inputs,
        inputs=[
            gr.Image(type="filepath", label="Upload an Image"),
            # gr.Checkbox(label="Extract text from image"),
            gr.Textbox(label="Plese input the requirements"),
            # gr.Textbox(label="Enter Second Text"),
            # gr.Textbox(label="Enter Third Text")
        ],
        outputs=[
            gr.Image(label="Matched Image"),
            gr.Textbox(label="The description of matched Airbnb resources"),
            gr.Textbox(label="The link of matched Airbnb resources"),
        ],
        title="Airbnb services",
        description=""
    )

    # Launch the app
    iface.launch()
