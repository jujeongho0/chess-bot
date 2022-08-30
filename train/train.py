import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, IntervalStrategy, Trainer, TrainingArguments


torch.manual_seed(42)
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
)

configuration = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)

model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda")
model.cuda()

f = open("chess_dataset.txt", "r")

while True:
    line = f.readline()
    break

f.close()


class PGNDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer(
                "<|startoftext|> " + txt[1:-15] + " <|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


dataset = PGNDataset(line.split("<|startoftext|>")[1:], tokenizer, max_length=1024)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    logging_steps=6000,
    save_strategy=IntervalStrategy.NO,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
)

Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda data: {
        "input_ids": torch.stack([f[0] for f in data]),
        "attention_mask": torch.stack([f[1] for f in data]),
        "labels": torch.stack([f[0] for f in data]),
    },
).train()

model.save_pretrained("gpt2.pth")

torch.manual_seed(42)
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>"
)

configuration = GPT2Config.from_pretrained("gpt2.pth", output_hidden_states=False)

model = GPT2LMHeadModel.from_pretrained("gpt2.pth", config=configuration)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda")

model.cuda()

generated = tokenizer("<|startoftext|> 1. e4", return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(
    generated,
    do_sample=True,
    top_k=50,
    bos_token="<|startoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
    max_length=300,
    top_p=0.95,
    temperature=1.9,
    num_return_sequences=20,
)
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
