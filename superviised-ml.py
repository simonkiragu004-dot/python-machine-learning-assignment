import numpy as np
from sklearn.linear_model import LinearRegression

# prepare training 
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# label
exam_scores = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

X = np.array(hours_studied).reshape(-1, 1)
y = np.array(exam_scores)
# create a linear regression model
model = LinearRegression()
# train the model
# Allow the user to enter the number of hours studied using input()
while True:
    user_input = input("Enter hours studied: ")
    try:
        hours = float(user_input)
        break
    except ValueError:
        print("Invalid input. Please enter a valid number.")
model.fit(X, y)
# prediction
predicted_exam_score = 72
prediction = model.predict(np.array([[hours]]))[0]
print(f"Predicted exam score for {hours} hours of study: {prediction}")
print(f"Actual exam score for {hours} hours of study: {predicted_exam_score}")
print(f"Predicted exam score: {prediction:.2f}")
import matplotlib.pyplot as plt

plt.scatter(hours_studied, exam_scores, color='blue', label='Data points')
plt.plot(hours_studied, model.predict(X), color='red', label='Regression line')
plt.xlabel('Hours studied')
plt.ylabel('Exam score')
plt.title('Linear Regression: Hours studied vs Exam score')
plt.legend()
plt.show()
