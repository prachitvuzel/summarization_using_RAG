import os
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


WIKIPEDIA_JSONL   = "/kaggle/input/rag-dataset/finetuning_dataset_wikipedia.jsonl"
GUARDIAN_JSONL    = "/kaggle/input/rag-dataset/finetuning_dataset_the_guardian.jsonl"


MODEL_LOCAL_PATH  = "/kaggle/input/mistral-7b-instruct/mistral-7b-instruct-v0.2"
MODEL_HF_PATH     = "mistralai/Mistral-7B-Instruct-v0.2"


PROCESSED_DATASET = "/kaggle/working/processed_dataset"
OUTPUT_DIR        = "/kaggle/working/rag-mistral-finetuned"
FINAL_MODEL_DIR   = "/kaggle/working/rag-mistral-finetuned/final"

def format_prompt(example):
    return {
        "text": f"""<s>[INST] {example['instruction']}

Context:
{example['input']} [/INST]

{example['output']}</s>"""
    }

if os.path.exists(PROCESSED_DATASET):
    
    print("Loading pre-processed dataset from disk...")
    dataset = load_from_disk(PROCESSED_DATASET)
    print(f"Dataset loaded instantly!")
    print(f"Train examples : {len(dataset['train'])}")
    print(f"Test examples  : {len(dataset['test'])}")

else:
   
    print("Processing dataset for first time...")

    wikipedia_dataset = load_dataset("json", data_files=WIKIPEDIA_JSONL)["train"]
    guardian_dataset  = load_dataset("json", data_files=GUARDIAN_JSONL)["train"]

    print(f"Wikipedia examples : {len(wikipedia_dataset)}")
    print(f"Guardian examples  : {len(guardian_dataset)}")

    wikipedia_sampled = wikipedia_dataset.shuffle(seed=42).select(range(min(20_000, len(wikipedia_dataset))))
    guardian_sampled  = guardian_dataset.shuffle(seed=42).select(range(min(20_000, len(guardian_dataset))))

    print(f"Sampled Wikipedia : {len(wikipedia_sampled)}")
    print(f"Sampled Guardian  : {len(guardian_sampled)}")

 
    combined_dataset = concatenate_datasets([wikipedia_sampled, guardian_sampled])
    combined_dataset = combined_dataset.shuffle(seed=42)

    dataset = combined_dataset.train_test_split(test_size=0.1)

 
    dataset = dataset.map(format_prompt)

    print(f"Train examples : {len(dataset['train'])}")
    print(f"Test examples  : {len(dataset['test'])}")

  
    dataset.save_to_disk(PROCESSED_DATASET)
    print(f"Dataset saved to {PROCESSED_DATASET}")


print("\nSample training example:")
print(dataset["train"][0]["text"])


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


if os.path.exists(MODEL_LOCAL_PATH):
    print(f"\nLoading model from Kaggle local path...")
    model_id = MODEL_LOCAL_PATH       
else:
    print(f"\nLocal model not found — downloading from HuggingFace...")
    print("This will take ~20-30 mins on first run...")
    model_id = MODEL_HF_PATH          

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
print("Model loaded!")


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded!")


model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=100,                  
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    report_to="none",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)


def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    return max(checkpoints, key=os.path.getctime) if checkpoints else None

latest_checkpoint = get_latest_checkpoint(OUTPUT_DIR)

if latest_checkpoint:
    print(f"\nResuming from : {latest_checkpoint}")
else:
    print("\nNo checkpoint found — starting fresh")


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_args,
)

trainer.train(resume_from_checkpoint=latest_checkpoint)


trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"\nTraining complete!")
print(f"Model saved to {FINAL_MODEL_DIR}")
