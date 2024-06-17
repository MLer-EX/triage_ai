from datasets import Dataset
from write_data import diagnosis_df


# آماده‌سازی داده‌ها
def preprocess_data(df):
    df = df.dropna()  # حذف مقادیر خالی
    df = df[['text', 'label']]  # انتخاب ستون‌های مورد نیاز
    df['label'] = df['label'].astype(int)  # تبدیل مقادیر label به عددی
    return df


diagnosis_df = preprocess_data(diagnosis_df)

# تبدیل به Dataset
dataset = Dataset.from_pandas(diagnosis_df)

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
