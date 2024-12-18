import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from datasets import load_from_disk
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling
from itertools import chain

# Load the dataset
dataset = load_from_disk('bookcorpus/chunked_ds')

# Define the tokenizer and data collator
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Define model configuration and initialize the model
configuration = GPT2Config(
    n_head=6,
    n_embd=450,
    n_layer=8,
)
model = GPT2LMHeadModel(configuration).to("cuda" if torch.cuda.is_available() else "cpu")

print(f"Model has {model.num_parameters():,} parameters.")

# Training parameters
num_train_epochs = 1
batch_size = 4
learning_rate = 2.5e-4
warmup_ratio = 0.05
weight_decay = 0.01
logging_steps = 50
save_steps = 10

# Prepare the DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)

# Define the optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay
)

# Learning rate scheduler
num_training_steps = num_train_epochs * len(data_loader)
warmup_steps = int(num_training_steps * warmup_ratio)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.train()

step = 0
for epoch in range(num_train_epochs):
    for batch in data_loader:
        # Move data to the appropriate device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Logging
        if step % logging_steps == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

        # Save model checkpoint
        if step % save_steps == 0 and step > 0:
            checkpoint_path = f"gpt-2-warm-up/40m/checkpoint-{step}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

        step += 1

# Final save
model.save_pretrained("gpt-2-warm-up/40m/final")
tokenizer.save_pretrained("gpt-2-warm-up/40m/final")