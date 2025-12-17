import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/student-mat.csv', sep=';')
df.head()


df.shape
df.columns
df.info()


df.isna().sum()

plt.hist(df['G3'])
plt.title("Final Grade Distribution")
plt.xlabel('Final Grade Distribution')
plt.ylabel('Number of Students')
plt.show()