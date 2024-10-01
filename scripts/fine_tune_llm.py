import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

# Custom dataset class for handling tokenization and formatting
class MedicalReportsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize input text (report name + history + observation)
        encodings = self.tokenizer(
            self.texts[idx], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        # Retrieve label (already converted to integer)
        label = self.labels[idx]
        
        # Return tokenized inputs and corresponding label
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)  # Ensure label is a long tensor
        }

# Preparing the dataset for tokenization
def prepare_data(df, label_encoder):
    # Concatenate 'Report Name', 'History', and 'Observation'
    texts = (df['Report Name'] + ' ' + df['History'] + ' ' + df['Observation']).tolist()
    # Transform string labels to numeric
    labels = label_encoder.fit_transform(df['Impression'].tolist())  # Convert labels to integers
    return texts, labels

# Load dataset
data_path = '../data/impression_300_llm.csv'
df = pd.read_csv(data_path)

# Split the data into training and evaluation
train_data = df[:300]
eval_data = df[300:]

# Initialize label encoder
label_encoder = LabelEncoder()

# Prepare training and evaluation data
train_texts, train_labels = prepare_data(train_data, label_encoder)
eval_texts, eval_labels = prepare_data(eval_data, label_encoder)

# Tokenizer and model
model_name = 'free-ai-ltd/ja-aozora-wikipedia-gamma-2b-chat'  # Adjust to your model
token = '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'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

# Create instances of the custom dataset
train_dataset = MedicalReportsDataset(train_texts, train_labels, tokenizer)
eval_dataset = MedicalReportsDataset(eval_texts, eval_labels, tokenizer)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='../results/checkpoints',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    logging_dir='../results/logs',
    logging_steps=10,
    save_steps=50,
)

# Fine-tuning the model using the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
