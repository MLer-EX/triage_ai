from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# بارگذاری دیتاست Iris
iris = load_iris()
X = iris.data
y = iris.target

# نمایش ویژگی‌ها و کلاس‌ها
print("Features:", iris.feature_names)
print("Target classes:", iris.target_names)



# تبدیل دیتاست به DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# تبدیل مقادیر کلاس‌ها به نام‌های گونه‌ها
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# نمایش چند سطر اول دیتاست
print(df.head())

# رسم نمودار جعبه‌ای برای تحلیل ویژگی‌ها
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient='h')
plt.title('Boxplot of Iris Dataset Features')
plt.show()

# رسم نمودار جفتی برای تحلیل همبستگی ویژگی‌ها
sns.pairplot(df, hue='species')
plt.show()
