from transformers import AutoTokenizer, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
print(f"Loaded initial tokenizer with {len(tokenizer)} tokens")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
tokenizer.pad_token = "!"
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1

# Add special tokens
special_tokens_dict = {'additional_special_tokens': ['[API]', '[/API]', 'Add', 'Mul']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))  # Also resize the token embeddings of the model
print(f"Added {num_added_toks} of new tokens to the tokenizer, adding up to {len(tokenizer)}")

config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=[ "w1", "w2", "w3"], lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, config)
# todo add new dataset
# dataset = load_dataset("harpreetsahota/modern-to-shakesperean-translation")
#print("dataset", dataset)
train_data = dataset["train"]

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
