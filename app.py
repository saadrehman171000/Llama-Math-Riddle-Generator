import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import torch
from safetensors.torch import load_file
import json
from torch.cuda.amp import autocast
import os

# Set environment variable to help manage memory allocation
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress TensorFlow warnings

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

# Force the model to run on CPU if GPU is not available or out of memory
device = "cpu"  # Force the model to run on the CPU for large models
model.to(device)

# Clear GPU memory after model load to avoid fragmentation (if using GPU)
torch.cuda.empty_cache()

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

    # Use mixed precision to save memory during inference
    with autocast():
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_length=50,  # Reduced max length for memory management
                num_beams=5,
                no_repeat_ngram_size=2,
                temperature=0.7,  # This will now work as we enable sampling
                top_p=0.9,        # This will now work as we enable sampling
                top_k=50,         # This will now work as we enable sampling
                do_sample=True    # Enable sampling
            )

    generated_riddle = tokenizer.decode(output[0], skip_special_tokens=True)
    st.subheader("Generated Riddle:")
    st.write(generated_riddle)

# If user inputs a riddle, generate the answer
if riddle_input:
    # Tokenize the input and generate an answer using the model
    inputs = tokenizer.encode(f"Question: {riddle_input} Answer:", return_tensors="pt").to(device)

    # Use mixed precision to save memory during inference
    with autocast():
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_length=50,  # Reduced max length for memory management
                num_beams=5,
                no_repeat_ngram_size=2,
                temperature=0.7,  # This will now work as we enable sampling
                top_p=0.9,        # This will now work as we enable sampling
                top_k=50,         # This will now work as we enable sampling
                do_sample=True    # Enable sampling
            )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    st.subheader("Generated Answer:")
    st.write(answer)

# Optionally, add a button to clear input
if st.button("Clear"):
    riddle_input = ""
    st.experimental_rerun()

# Clear GPU memory after inference
torch.cuda.empty_cache()
