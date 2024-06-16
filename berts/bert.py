from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from procces_data import train_dataset, test_dataset

# توکنایزر و مدل
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# توکنایز کردن داده‌ها
def preprocess_data(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)


train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

# تنظیمات آموزش
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# آموزش مدل
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
