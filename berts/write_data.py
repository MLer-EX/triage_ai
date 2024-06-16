import pandas as pd
from oss import new_directory
# خواندن داده‌ها از فایل‌های CSV
diagnosis_df = pd.read_csv(new_directory/'data/diagnosis.csv')
edstays_df = pd.read_csv(new_directory/'data/edstays.csv')
medrecon_df = pd.read_csv(new_directory/'data/medrecon.csv')
pyxis_df = pd.read_csv(new_directory/'data/pyxis.csv')
triage_df = pd.read_csv(new_directory/'data/triage.csv')
vitalsign_df = pd.read_csv(new_directory/'data/vitalsign.csv')

# نمایش نمونه‌ای از داده‌ها
print(diagnosis_df.head())
print(edstays_df.head())
print(medrecon_df.head())
print(pyxis_df.head())
print(triage_df.head())
print(vitalsign_df.head())
