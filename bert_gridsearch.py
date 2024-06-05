
!pip install transformers torch optuna
!pip install accelerate -U



import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import optuna

df = pd.read_csv("path_to_dataset")

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


model_name = 'dbmdz/bert-base-turkish-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
num_labels = len(df['labels'].unique())
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


train_dataset = ConversationDataset(train_df['conversation'].tolist(), train_df['labels'].tolist(), tokenizer, max_length=512)
val_dataset = ConversationDataset(val_df['conversation'].tolist(), val_df['labels'].tolist(), tokenizer, max_length=512)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {'f1': f1, 'accuracy': accuracy}


def objective(trial):
    training_args = TrainingArguments(
        output_dir='./bert_fine_tuned_model',
        num_train_epochs=trial.suggest_int('num_train_epochs', 1, 5),
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32]),
        per_device_eval_batch_size=8,
        warmup_steps=trial.suggest_int('warmup_steps', 0, 500),
        weight_decay=trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
        logging_dir='./logs',
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=100,
    )

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result['eval_f1']


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial

print(f"  F1 Score: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


best_params = trial.params
training_args = TrainingArguments(
    output_dir='./bert_fine_tuned_model',
    num_train_epochs=best_params['num_train_epochs'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    per_device_eval_batch_size=8,
    warmup_steps=best_params['warmup_steps'],
    weight_decay=best_params['weight_decay'],
    logging_dir='./logs',
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=100,
)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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
