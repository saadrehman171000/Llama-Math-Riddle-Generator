import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import torch
from safetensors.torch import load_file
import json

# Define paths for your model and adapter files
model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Pre-trained TinyLlama model from Hugging Face
adapter_model_path = "fine-tuned-QA-tinyllama-1.1B/adapter_model.safetensors"  # LoRA model weights
adapter_config_path = "fine-tuned-QA-tinyllama-1.1B/adapter_config.json"  # LoRA adapter config

# Load pre-trained TinyLlama model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path)

# Load the LoRA adapter configuration using json module
with open(adapter_config_path, 'r') as f:
    adapter_config = json.load(f)

# Initialize LoraConfig using the necessary parameters
lora_config = LoraConfig(
    r=adapter_config.get("r", 16),  # Low-rank adaptation size
    lora_alpha=adapter_config.get("lora_alpha", 32),  # LoRA scaling factor
    target_modules=adapter_config.get("target_modules", ["q_proj", "v_proj"]),  # Targeted modules for LoRA
    lora_dropout=adapter_config.get("lora_dropout", 0.05),  # Dropout for LoRA layers
    bias="none",  # Bias configuration
    base_model_name_or_path=model_path  # Base model path
)

# Apply LoRA to the pre-trained base model using PEFT (Low-Rank Adaptation)
model = get_peft_model(base_model, lora_config)

# Load the LoRA fine-tuned weights using safetensors
model.load_state_dict(load_file(adapter_model_path), strict=False)  # Using strict=False to avoid missing key errors

# Set the model to use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Streamlit UI setup
st.title("Math Riddle Generator and Factory")
st.write(
    "This app generates answers for math riddles using a fine-tuned TinyLlama model. You can input a riddle, and the model will generate an answer, or you can generate new riddles using the 'Generate Riddle' button."
)

# Input: Text box for user to input a riddle
riddle_input = st.text_input("Enter a math riddle:")

# Generate a new riddle using the fine-tuned model
if st.button("Generate a New Math Riddle"):
    # Generate a new riddle using the model
    prompt = "Generate a new math riddle:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=100,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

    generated_riddle = tokenizer.decode(output[0], skip_special_tokens=True)
    st.subheader("Generated Riddle:")
    st.write(generated_riddle)

# If user inputs a riddle, generate the answer
if riddle_input:
    # Tokenize the input and generate an answer using the model
    inputs = tokenizer.encode(f"Question: {riddle_input} Answer:", return_tensors="pt").to(device)

    # Generate the answer
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=100,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    st.subheader("Generated Answer:")
    st.write(answer)

# Optionally, add a button to clear input
if st.button("Clear"):
    riddle_input = ""
    st.experimental_rerun()
