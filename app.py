# Write code for importing necessary libraries

import gradio as  gr
from transformers import BlipProcessor , BlipForConditionalGeneration
from PIL import Image
import torch

import warnings
warnings.filterwarnings('ignore')

# Write code for developing image captioning model

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Write code for developing image captioning function

def generate_caption(image):
    """Generate a caption for the uploaded image"""
    raw_image = Image.open(image).convert("RGB")
    inputs = processor(raw_image,return_tensors='pt')
    with torch.no_grad():
        out =  model.generate(**inputs)  
        caption = processor.decode(out[0],skip_special_tokens=True)
        return caption

# Write code for developing image captioning interface

iface = gr.Interface(
    fn = generate_caption,
    inputs = gr.Image(type = "filepath"),
    outputs = gr.Textbox(label = "Generated Caption"),
    title = "AI Image Captioning",
    description =  "Upload an image, and the AI will generate a descriptive caption for it using the BLIP model."
)

# Write code for launching the app

if __name__ == "__main__":
    iface.launch(share=True)

