import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# Load model once
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "AIDC-AI/Ovis1.6-Gemma2-9B",
        torch_dtype=torch.bfloat16,
        multimodal_max_length=8192,
        trust_remote_code=True
    ).cuda()
    return model

model = load_model()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# Streamlit app layout
st.title("Image and Text Prompt Generator")
st.write("Upload an image and enter a prompt to generate text output.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Text prompt input
text = st.text_input("Enter your prompt:")

if st.button("Generate Output"):
    if uploaded_file and text:
        # Open image
        image = Image.open(uploaded_file)
        
        # Prepare input
        query = f'<image>\n{text}'
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

        # Generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

        # Display output
        st.write(f'**Output:**\n{output}')
    else:
        st.error("Please upload an image and enter a prompt.")