# LLM-Assisted University Timetable Generation

This project demonstrates how to build an **LLM-assisted timetable generation system** using **historical UniTime XML schedules**. It covers **data preparation, dataset creation, and QLoRA-based fine-tuning** of instruction-tuned large language models.

---

## 📂 Repository Contents

- **`Hafsa_TimeTable_1.ipynb`** → Initial XML parsing & exploratory analysis.  
- **`cs_goal_1 (1).ipynb`** → Goal 1: Convert UniTime XML → JSONL datasets.  
- **`goal_2.ipynb`** → Goal 2: Dataset structuring for fine-tuning.  
- **`Hafsa_Time_Table_Goal2_updated.ipynb`** → Improved Goal 2 pipeline.  

---

## 🎯 Project Goals

1. **Goal 1 – Data Preparation**  
   - Parse UniTime XML files.  
   - Convert into JSONL datasets for LLM training.  

2. **Goal 2 – LLM Fine-Tuning (QLoRA)**  
   - Train using Hugging Face **Transformers + PEFT (QLoRA)**.  
   - Generate feasible timetables per class and per offering.  

3. **Goal 3 – Constraint Validation (Planned)**  
   - Validate against room capacity, overlaps, and instructor availability.  
   - Auto-repair minor violations.  

---

## 📑 Dataset Format

Each example is instruction-style JSONL:

```json
{"instruction": "Assign room and time slot",
 "input": "Course: CS101, Instructor: Dr. A, Room Capacity: 50",
 "output": "Monday 9-11 AM, Room 201"}
pip install -q "transformers==4.44.2" "peft==0.12.0" "accelerate>=0.33.0" "bitsandbytes==0.43.1" datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# 1. Load dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl", "val": "val.jsonl"})

# 2. Load base model in 4-bit
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. Apply LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Training setup
args = TrainingArguments(
    output_dir="./qlora-timetable",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

# 5. Tokenize
def preprocess(example):
    text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(preprocess)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"]
)

trainer.train()
prompt = "Course: AI101, Instructor: Dr. Smith, Needs Lab with 40 seats"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
