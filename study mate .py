!pip install --upgrade transformers torch torchvision gradio accelerate

import torch
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration
)

# ----------------------------------------------------
# TEXT CHATBOT (IBM GRANITE 3.3 2B)
# ----------------------------------------------------

model_name = "ibm-granite/granite-3.3-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
)

def chat(message, history):
    prompt = f"User: {message}\nAssistant:"
    response = chatbot(prompt)[0]["generated_text"]

    # extract only assistant reply
    if "Assistant:" in response:
        response = response.split("Assistant:")[1].strip()

    return response


# ----------------------------------------------------
# IMAGE CAPTIONING (BLIP)
# ----------------------------------------------------

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def analyze(image):
    inputs = processor(image, return_tensors="pt")
    out = blip.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# ----------------------------------------------------
# GRADIO UI
# ----------------------------------------------------

with gr.Blocks() as ui:
    gr.Markdown("## 🔥 Granite Chat + Image Captioning App (Error-Free)")

    with gr.Tab("Chat"):
        msg = gr.Textbox(label="Say something")
        reply = gr.Textbox(label="Response")
        gr.Button("Send").click(chat, msg, reply)

    with gr.Tab("Image Caption"):
        img = gr.Image(type="pil")
        cap = gr.Textbox(label="Caption")
        gr.Button("Analyze").click(analyze, img, cap)

ui.launch()