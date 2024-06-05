
!pip install transformers torch
!pip install accelerate -U
!pip install turkish-lm-tuner

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv("path_to_dataset")
df['length'] = df['conversation'].apply(lambda s: len(s))
df['length'].describe()

def convert_labels(label):
    if label in [-1, 0, 1]:
        return 0
    elif label == 2:
        return 1
    else:
        return label


df['labels'] = df['labels'].apply(convert_labels)

df = df.drop(columns=['row_count'])


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

class ConversationDataset(Dataset):
    def __init__(self, text_list, label_list, tokenizer, max_length):
        self.text_list = text_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = str(self.text_list[idx])
        label = int(self.label_list[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


from turkish_lm_tuner import T5ForClassification
from transformers import AutoConfig
model_name = 'boun-tabi-LMG/TURNA'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = len(df['labels'].unique())
model = T5ForClassification(model_name, config, num_labels=num_labels, problem_type='single_label_classification')

train_dataset = ConversationDataset(train_df['conversation'].tolist(), train_df['labels'].tolist(), tokenizer, max_length=550)
val_dataset = ConversationDataset(val_df['conversation'].tolist(), val_df['labels'].tolist(), tokenizer, max_length=550)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {'f1': f1, 'accuracy': accuracy}


training_args = TrainingArguments(
    output_dir='./turna_fine_tuned_model',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.001,
    logging_dir='./logs',
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()


results = trainer.evaluate()
print(results)

print("F1 Score:", results['eval_f1'])
print("Accuracy:", results['eval_accuracy'])
