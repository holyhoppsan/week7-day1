import torch
from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#peft_model_id = "mixtral-moe-lora-instruct-shapeskeare/checkpoint-50"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Tokens to check
special_tokens = ["<s>", "</s>", "[INST]", "[/INST]", "[API]", "[/API]"]

# Check each token
for token in special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:
        print(f"Token {token} is not recognized by the tokenizer.")
    else:
        print(f"Token {token} is recognized by the tokenizer and has an ID of {token_id}.")


model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")
# model.load_adapter(peft_model_id)
model.eval()

# Set up the text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/generate/")
async def call_generation_function(input_text: str):
    return generate_shakespearean_text(input_text)

def generate_shakespearean_text(input_text: str):
    if not input_text:
        raise ValueError("Input text is required")  # Changed from HTTPException for direct script execution compatibility

    sys_msg = f"Generate the code for a valid C# script that is compatible with Unity, and that carries out the action or task requested by the prompt I will provide on the next line, I only want the code no other wording or explanation apart from the code. \n {input_text}"
    #prompt = f"<s> [INST]{sys_msg}\n{input_text}[/INST]</s>"
    prompt = f"{sys_msg} {input_text}"

    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=1000)

    notes = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"New: original_text": input_text, "shakespearean_text": notes}
    
    # sys_msg = "Translate the given text to Shakespearean style."
    # prompt = f"<s> [INST]{sys_msg}\n{input_text}[/INST]</s>"
    # results = text_generator(prompt, max_length=512, num_return_sequences=1)
    # return {"original_text": input_text, "shakespearean_text": results[0]['generated_text']}

def main():
    input_text = "Add 2 and 3 to a variable called hej"
    result = generate_shakespearean_text(input_text)
    print(result)

if __name__ == "__main__":
    main()
