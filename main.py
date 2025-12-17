import pandas as pd
import matplotlib.pyplot as plt

''' 
Goal: Analyze which academic and 
lifestyle factors are most associated with students' final grades (G3)
'''


df = pd.read_csv('data/student-mat.csv', sep=';')
df.head()
df.shape
df.columns
df.info()


'''
The dataset contains 395 students and demographic, 
academic, and lifestyle attributes, 
along with first, second, and final grades.
'''

plt.hist(df['G3'], bins=10)
plt.title('Distribution of Final Grades')
plt.xlabel('Final Grade')
plt.ylabel('Number of Students')
plt.show()

'''
Most students receive mid-range grades, with fewer students scoring very low or very high.
'''

# Study Time vs Perfomance
studytime = df.groupby('studytime')['G3'].mean()

studytime.plot(kind='bar')
plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time Level")
plt.ylabel("Average Final Grade")
plt.show()

'''
Students who report higher study time generally achieve higher average 
final grades, suggesting a positive association between study habits and performance.

'''

failures = df.groupby("failures")["G3"].mean()
plt.scatter(df["failures"], df["G3"])
plt.title("Past Failures vs Final Grade")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade")
plt.show()


'''
Prior academic failures are strongly negatively associated with final grades.
'''


plt.scatter(df["absences"], df["G3"])
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.show()
df["absences"].corr(df["G3"])


'''
Absences show a weak to moderate negative correlation with final grades, indicating that attendance may impact academic outcomes but is not the sole determining factor.
'''


df.groupby("sex")["G3"].mean()
df.groupby("sex")["G3"].mean().plot(kind="bar")
plt.title("Average Final Grade by Gender")
plt.ylabel("Average Final Grade")
plt.show()

'''
Average performance differs slightly by gender, though the difference is relatively small and may not be statistically significant.
'''
df[["studytime", "failures", "absences", "G1", "G2", "G3"]].corr()["G3"].sort_values(ascending=False)


'''
Previous period grades (G1, G2) are the strongest predictors of final performance, followed by study time and past failures.
'''


'''
Conclusion:
Student performance is most strongly associated with prior academic outcomes and consistent study habits. While lifestyle factors such as absences and free time show some relationship with grades, academic history remains the dominant factor influencing final performance.
'''