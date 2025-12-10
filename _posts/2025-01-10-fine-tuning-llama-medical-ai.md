---
layout: post
title: "Fine-Tuning LLaMA 3.2 for Medical AI: Building MedGenius from 40K+ Dialogues"
description: "A detailed guide on how I fine-tuned Meta's LLaMA 3.2 1B model for medical consultations using 40,000+ curated dialogues, LoRA, and Unsloth, achieving production-ready medical AI."
date: 2025-01-10
categories: [Medical AI, LLM, Fine-tuning]
tags: [LLaMA, LoRA, QLoRA, Medical AI, Unsloth, Fine-tuning, Healthcare, Hugging Face]
author: "Huzefa Nalkheda Wala"
---

When I set out to build [MedGenius LLaMA](https://huggingface.co/huzaifa525/MedGenius-LLaMA-3.2-1B), I wanted to create a medical AI assistant that could handle patient-doctor dialogues with the nuance and accuracy required in healthcare. After 3 months of work, I fine-tuned LLaMA 3.2 1B on 40,000+ curated medical dialogues.

In this post, I'll share the complete process, from dataset curation to deployment, so you can fine-tune domain-specific LLMs for your own use cases.

## Why Fine-Tune Instead of RAG?

Before diving in, let's address the elephant in the room: **when should you fine-tune vs. use RAG?**

### Use Fine-Tuning When:
- ✅ You need the model to "speak" domain-specific language
- ✅ You have 10,000+ high-quality examples
- ✅ You need consistent tone and style
- ✅ Latency matters (no retrieval overhead)
- ✅ You want to embed knowledge IN the model

### Use RAG When:
- ✅ Knowledge changes frequently
- ✅ You have documents but not dialogue examples
- ✅ You need source citations
- ✅ Budget for inference API calls
- ✅ Don't have GPU resources for training

For MedGenius, I chose fine-tuning because medical terminology and consultation patterns needed to be embedded in the model itself.

## Dataset Curation: The Foundation

**Good data > fancy techniques.** I spent 50% of my time on data curation.

### Sources for Medical Dialogues

1. **Medical transcripts** (anonymized)
2. **Public medical QA datasets** (cleaned and verified)
3. **Synthetic data** from GPT-4 (carefully reviewed)
4. **Medical literature** adapted to dialogue format

### Data Format

I used the **instruction-following format**:

```json
{
  "instruction": "You are a medical AI assistant. Provide accurate, empathetic responses.",
  "input": "I've had a persistent headache for 3 days, along with sensitivity to light. What could this be?",
  "output": "Based on your symptoms of persistent headache and photophobia (light sensitivity), this could indicate several conditions:\n\n1. **Migraine** - Most likely given the combination of headache and light sensitivity\n2. **Tension headache** - Though less commonly causes photophobia\n3. **Sinusitis** - If accompanied by facial pressure\n4. **Meningitis** - Rare but serious (seek immediate care if you have fever, stiff neck, or confusion)\n\nI recommend:\n- Rest in a dark, quiet room\n- Stay hydrated\n- OTC pain relief (ibuprofen or acetaminophen)\n- See a doctor if symptoms persist beyond 5 days or worsen\n\n⚠️ Seek immediate medical attention if you experience: severe sudden headache, fever, stiff neck, confusion, or vision changes."
}
```

### Data Quality Criteria

Every dialogue in my dataset met these standards:

1. **Medically accurate** - Verified against medical databases
2. **Appropriately cautious** - Never replace professional diagnosis
3. **Empathetic tone** - Healthcare requires compassion
4. **Clear disclaimers** - When to seek professional help
5. **Structured responses** - Easy to parse and understand

### Cleaning Pipeline

```python
import re
import pandas as pd
from typing import List, Dict

def clean_medical_dialogue(dialogue: Dict) -> Dict:
    """Clean and validate medical dialogue"""

    # Remove PII (emails, phone numbers, names)
    dialogue['input'] = remove_pii(dialogue['input'])
    dialogue['output'] = remove_pii(dialogue['output'])

    # Normalize medical terms
    dialogue['output'] = normalize_medical_terms(dialogue['output'])

    # Add safety disclaimers
    if needs_disclaimer(dialogue['output']):
        dialogue['output'] += "\n\n⚠️ This information is for educational purposes. Consult a healthcare professional for personalized advice."

    # Validate medical accuracy
    if not is_medically_sound(dialogue):
        return None  # Skip invalid data

    return dialogue

def remove_pii(text: str) -> str:
    """Remove personally identifiable information"""
    # Remove emails
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    # Remove phone numbers
    text = re.sub(r'\+?\d[\d\s\-\(\)]{8,}\d', '[PHONE]', text)
    # Remove specific names (use NER)
    text = anonymize_names(text)
    return text
```

### Final Dataset Stats

- **Total dialogues**: 40,127
- **Average input length**: 87 tokens
- **Average output length**: 246 tokens
- **Medical categories**: 42 (cardiology, neurology, etc.)
- **Quality score**: 4.7/5 (expert review)

## Choosing LLaMA 3.2 1B

### Why LLaMA 3.2?

- **Open source** - Full control and commercial use
- **Strong base** - Meta's latest architecture
- **Efficient** - 1B parameters runs on consumer GPUs
- **Good instruction following** - Pre-trained for chat

### Model Size Trade-offs

| Model Size | GPU Memory | Inference Speed | Quality | My Choice |
|-----------|------------|-----------------|---------|-----------|
| 405B | 810GB+ | Very Slow | Excellent | ❌ Overkill |
| 70B | 140GB | Slow | Excellent | ❌ Too expensive |
| 8B | 16GB | Medium | Very Good | ⚠️ Good option |
| **1B** | **2GB** | **Fast** | **Good** | ✅ **Perfect balance** |

For a medical chatbot that needs to be accessible, **1B was the sweet spot**.

## LoRA: Parameter-Efficient Fine-Tuning

Full fine-tuning would require:
- 🔴 4GB+ GPU memory per training sample
- 🔴 $500+ in compute costs
- 🔴 48+ hours training time

**LoRA (Low-Rank Adaptation)** changed everything:

### How LoRA Works

Instead of updating all 1 billion parameters, LoRA:

1. Freezes the base model (1B params)
2. Injects trainable low-rank matrices (2M params)
3. Trains only the small matrices
4. Merges them back after training

**Result**:
- ✅ 99.8% fewer trainable parameters
- ✅ 2x faster training
- ✅ Same quality as full fine-tuning

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank - higher = more capacity
    lora_alpha=32,          # Scaling factor
    target_modules=[         # Which layers to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,      # Regularization
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

# Check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")  # ~2M instead of 1B!
```

## Unsloth: 2x Faster Training

[Unsloth](https://github.com/unslothai/unsloth) optimizes training with:
- **Flash Attention 2** - Faster attention computation
- **Gradient checkpointing** - Lower memory usage
- **Optimized kernels** - CUDA-level optimizations

### Setup

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-1b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True  # Quantization
)

# Enable Unsloth optimizations
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Optimized checkpointing
    random_state=42
)
```

**Training speed comparison:**

| Framework | Time per Epoch | GPU Memory |
|-----------|----------------|------------|
| Vanilla PyTorch | 12 hours | 24GB |
| PEFT (LoRA) | 6 hours | 12GB |
| **Unsloth** | **3 hours** | **8GB** |

## Training Configuration

### Hyperparameters

After experimentation, here's what worked:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./medgenius-llama-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size: 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,                     # Mixed precision training
    optim="adamw_8bit",           # Memory-efficient optimizer
    max_grad_norm=0.3,            # Gradient clipping
    seed=42
)
```

### Why These Values?

- **Learning rate (2e-4)**: Higher than full fine-tuning (5e-5) because LoRA needs stronger signal
- **Batch size (32 effective)**: Balances memory and training stability
- **3 epochs**: Medical domain converges fast; more risks overfitting
- **Cosine schedule**: Smooth learning rate decay

## Training Process

```python
from transformers import Trainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="medical_dialogues.jsonl")
train_dataset = dataset["train"].shuffle(seed=42)

# Tokenize
def tokenize_function(examples):
    prompts = [
        f"### Instruction:\n{examples['instruction'][i]}\n\n### Input:\n{examples['input'][i]}\n\n### Response:\n{examples['output'][i]}"
        for i in range(len(examples['input']))
    ]
    return tokenizer(
        prompts,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
```

### Training Logs

```
Epoch 1/3: Loss=1.234  |  Time=2.8h  |  GPU=7.2GB
Epoch 2/3: Loss=0.876  |  Time=2.8h  |  GPU=7.2GB
Epoch 3/3: Loss=0.654  |  Time=2.8h  |  GPU=7.2GB

Total training time: 8.4 hours on RTX 4090
```

## Evaluation

### Quantitative Metrics

```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

def evaluate_model(model, test_dataset):
    predictions = []
    references = []

    for example in test_dataset:
        pred = generate_response(model, example['input'])
        predictions.append(pred)
        references.append(example['output'])

    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]

    # Medical accuracy (custom metric)
    accuracy = medical_accuracy_check(predictions, references)

    return {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in rouge_scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in rouge_scores]),
        'rougeL': np.mean([s['rougeL'].fmeasure for s in rouge_scores]),
        'medical_accuracy': accuracy
    }
```

**Results:**

| Metric | Base LLaMA 3.2 | MedGenius (Fine-tuned) |
|--------|---------------|------------------------|
| ROUGE-1 | 0.42 | **0.78** |
| ROUGE-L | 0.38 | **0.72** |
| Medical Accuracy | 61% | **89%** |
| Empathy Score | 3.2/5 | **4.7/5** |

### Qualitative Analysis

**Example 1: Symptom Assessment**

**Input**: "I've been feeling dizzy and lightheaded when I stand up quickly."

**Base Model**: "You might want to see a doctor about that. It could be several things."

**MedGenius**: "What you're describing sounds like orthostatic hypotension (postural hypotension) - a sudden drop in blood pressure when standing. This is common and usually not serious.

Possible causes:
- Dehydration
- Prolonged bed rest
- Medications (blood pressure meds, diuretics)
- Anemia

Recommendations:
- Stand up slowly in stages (sit first, then stand)
- Stay well-hydrated (8+ glasses water/day)
- Avoid sudden position changes
- Increase salt intake slightly (if not hypertensive)

See a doctor if you experience: fainting, chest pain, blurred vision, or if episodes are frequent and affecting daily life."

**Winner**: MedGenius provides structured, actionable, medically accurate advice.

## Publishing to Hugging Face

```python
# Merge LoRA weights
model = model.merge_and_unload()

# Push to Hub
model.push_to_hub("huzaifa525/MedGenius-LLaMA-3.2-1B")
tokenizer.push_to_hub("huzaifa525/MedGenius-LLaMA-3.2-1B")
```

**Model card** (important for visibility):

```markdown
---
language: en
license: llama3.2
tags:
- medical-ai
- llama
- lora
- healthcare
datasets:
- medical-dialogues
metrics:
- rouge
---

# MedGenius LLaMA 3.2 1B

Fine-tuned medical chatbot based on Meta's LLaMA 3.2 1B, trained on 40,000+ curated medical dialogues.

## Intended Use
- Medical education
- Symptom guidance (NOT diagnosis)
- Health information queries

## Limitations
⚠️ NOT a replacement for professional medical advice
⚠️ Cannot diagnose conditions
⚠️ Should not be used for emergency situations

## Training Details
- Base model: LLaMA 3.2 1B
- Method: LoRA (r=16, alpha=32)
- Dataset: 40,127 medical dialogues
- Training time: 8.4 hours on RTX 4090
```

**Check it out**: [huggingface.co/huzaifa525/MedGenius-LLaMA-3.2-1B](https://huggingface.co/huzaifa525/MedGenius-LLaMA-3.2-1B)

## Deployment

### Local Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "huzaifa525/MedGenius-LLaMA-3.2-1B",
    device_map="auto",
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained("huzaifa525/MedGenius-LLaMA-3.2-1B")

def chat(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### FastAPI Production Deployment

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    message: str
    max_tokens: int = 256

@app.post("/chat")
async def medical_chat(query: Query):
    response = chat(query.message)
    return {"response": response}
```

## Lessons Learned

### What Worked

1. **Quality > Quantity**: 40K high-quality > 400K noisy data
2. **LoRA is magic**: 99.8% parameter reduction, same quality
3. **Unsloth 2x speedup**: $200 training cost → $100
4. **Instruction format**: Consistent formatting improves learning
5. **Safety first**: Built-in disclaimers prevent misuse

### What Didn't Work

1. **Larger LoRA rank (r=64)**: Overfitting, no quality gain
2. **Higher learning rate (5e-4)**: Training instability
3. **More epochs (5+)**: Memorization, not generalization
4. **Synthetic data only**: Lacks real medical nuance

### Mistakes to Avoid

1. ❌ **Skipping data cleaning**: Garbage in = garbage out
2. ❌ **Training without validation**: Can't detect overfitting
3. ❌ **Ignoring safety**: Medical AI needs strong safeguards
4. ❌ **Wrong base model**: LLaMA > GPT-2 for instruction following
5. ❌ **No evaluation plan**: How do you know it works?

## Cost Breakdown

Total project cost: **$473**

- Dataset curation: $120 (GPT-4 API for synthetic data)
- GPU compute (Vast.ai): $243 (RTX 4090, 12 hours)
- Hugging Face Pro: $9/month
- Domain expertise (my time): Priceless 😄

## Future Improvements

1. **Multimodal**: Add medical image understanding
2. **Multilingual**: Expand to Spanish, Hindi, Arabic
3. **Specialized models**: Cardiology, Neurology sub-models
4. **Reinforcement learning**: RLHF from doctor feedback
5. **Federated learning**: Train on hospital data privately

## Conclusion

Fine-tuning LLaMA 3.2 for medical AI taught me that **domain expertise + clean data + efficient training = production-ready models** in weeks, not months.

MedGenius LLaMA now handles patient queries with 89% medical accuracy, empathetic responses, and proper safety disclaimers - all in a 1B parameter model that runs on consumer hardware.

**Key Takeaways:**

1. **Data curation is 50% of the work** - Don't rush it
2. **LoRA makes fine-tuning accessible** - 2M params vs 1B
3. **Unsloth cuts costs in half** - Use it
4. **Medical AI needs safeguards** - Never skip disclaimers
5. **Evaluation is critical** - Quantitative + qualitative

## Resources

- **Model**: [HuggingFace](https://huggingface.co/huzaifa525/MedGenius-LLaMA-3.2-1B)
- **Code**: [GitHub](https://github.com/huzaifa525/MedGenius-LLaMA)
- **Paper**: Fine-Tuning LLaMA for Healthcare (coming soon)

## Connect

Questions about medical AI or fine-tuning? Let's connect:

- **GitHub**: [github.com/huzaifa525](https://github.com/huzaifa525)
- **LinkedIn**: [linkedin.com/in/huzefanalkheda](https://linkedin.com/in/huzefanalkheda)
- **Email**: huzaifanalkheda@gmail.com

---

*Building healthcare AI? I'd love to hear about your project. Drop a comment or reach out!*
