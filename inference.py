import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
peft_model_id = "mixtral-moe-lora-instruct-shapeskeare/checkpoint-50"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")
model.load_adapter(peft_model_id)

# Set up the text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_shakespearean_text(input_text):
    # Use the trained model to translate text to Shakespearean style
    results = text_generator(f"Translate the given text to Shakespearean style. {input_text}", max_length=512, num_return_sequences=1)
    return results[0]['generated_text']

# Example usage
input_text = "What are you doing tomorrow?"
shakespearean_text = generate_shakespearean_text(input_text)
print(shakespearean_text)
