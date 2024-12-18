from itertools import chain
import warnings
import math

#from hf
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers import TrainingArguments, Trainer

# loading raw data
dataset = load_dataset("raddwolf/BookCorpus74M",trust_remote_code=True)

# make splits
dataset = dataset['train'].select(range(25000000))

dataset.train_test_split(test_size=0.0015) 

# load the gpt-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token=tokenizer.eos_token

# tokenize
def tokenize_function(example):
    return tokenizer(text=example["text"])
tokenized_ds = dataset.map(tokenize_function,batched=True,remove_columns='text')

# save to disk if required (use load_from_disk latter)
tokenized_ds.save_to_disk('bookcorpus/tokenized_ds')

# Make samples to a size of 1024
def concat(examples):    
    examples["input_ids"]=[list(chain.from_iterable(examples['input_ids']))] # convert chain to list of tokens
    examples["attention_mask"]=[list(chain.from_iterable(examples['attention_mask']))] # convert chain to list of tokens
    return examples
    
# takes a lot of time (worth saving it to disk)
concated_ds = tokenized_ds.map(concat,batched=True,batch_size=1000000,num_proc=8)

def chunk(examples):
    chunk_size = 1024 # modify this accordingly       
    input_ids = examples["input_ids"][0] # List[List], pass the inner list      
    attention_mask = examples["attention_mask"][0] # List[List]
    input_ids_truncated = []
    attention_mask_truncated = []
    
    #slice with step_size=chunk_size
    for i in range(0,len(input_ids),chunk_size):
        chunk = input_ids[i:i+chunk_size]
        if len(chunk)==chunk_size: # drop the last chunk if not equal
            input_ids_truncated.append(chunk)
            attention_mask_truncated.append(attention_mask[i:i+chunk_size])     
    examples['input_ids']=input_ids_truncated
    examples["attention_mask"]=attention_mask_truncated
        
    return examples   

chunked_ds = concated_ds.map(chunk,batched=True,batch_size=2,num_proc=2)
chunked_ds.save_to_disk('bookcorpus/chunked_ds') # will use this latter for diff experimentation

data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)

# load the model
configuration = GPT2Config(
    n_head=6,
    n_embd=450,
    n_layer=8,
)
model = GPT2LMHeadModel(configuration)
print(f"{model.num_parameters():,}")

# training arguments
training_args = TrainingArguments( output_dir='gpt-2-warm-up/40m',
                                  # evaluation_strategy="steps",
                                  # eval_steps=500,                                  
                                  # per_device_eval_batch_size=8,
                                  num_train_epochs=1,
                                  per_device_train_batch_size=4,
                                  learning_rate=2.5e-4,
                                  lr_scheduler_type='cosine',
                                  warmup_ratio=0.05,
                                  adam_beta1=0.9,
                                  adam_beta2=0.999,                                  
                                  weight_decay=0.01,                                  
                                  logging_strategy="steps",
                                  logging_steps = 50,
                                  save_steps=10,
                                  save_total_limit=10,                                  
                                 ) 
trainer = Trainer(model=model,
                 args = training_args,
                 tokenizer=tokenizer,
                 train_dataset=chunked_ds,
                 # eval_dataset=chunked_ds,
                 data_collator = data_collator)

trainer.train(resume_from_checkpoint=True)
#trainer.train(resume_from_checkpoint=False)
