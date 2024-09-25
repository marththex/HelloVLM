import streamlit as st
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from collections.abc import Generator as generator


# Load model and tokenizer once
@st.cache_resource
def load_model():
    model = AutoModel.from_pretrained(
        'openbmb/MiniCPM-V-2_6',
        trust_remote_code=True,
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app layout
st.title("Image Question Answering")
st.write("Upload an image and enter a question to generate a response.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Question input
question = st.text_input("Enter your question:")

if st.button("Generate Answer"):
    if uploaded_file and question:
        # Open image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Prepare message
        msgs = [{'role': 'user', 'content': [image, question]}]

        # Generate output
        with torch.inference_mode():
            # Perform chat
            res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
            
            # If streaming is enabled
            if isinstance(res, (list, generator)):  # Adjust based on actual return type
                generated_text = ""
                for new_text in res:
                    generated_text += new_text
                    st.write(new_text)  # Display each new text in the Streamlit app
            else:
                st.write(res)  # If not streaming, show the final result

    else:
        st.error("Please upload an image and enter a question.")
