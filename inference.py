import torch
from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
peft_model_id = "mixtral-moe-lora-instruct-shapeskeare/checkpoint-50"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")
model.load_adapter(peft_model_id)

# Set up the text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/generate/")
async def generate_shakespearean_text(input_text: str):
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is required")
    # Generate text
    results = text_generator(f"Translate the given text to Shakespearean style. {input_text}", max_length=512, num_return_sequences=1)
    return {"original_text": input_text, "shakespearean_text": results[0]['generated_text']}
