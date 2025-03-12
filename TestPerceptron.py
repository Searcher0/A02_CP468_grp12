import csv
from FromScratch import Perceptron as p

# Read CSV file and store data
data = []
with open('WeatherData_Q3.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        try:
            # parse columns as floats/integers
            temp = float(row[0])
            humid = float(row[1])
            label = int(row[2])  # 0 or 1
            data.append((temp, humid, label))
        except ValueError:
            # skip any rows that fail parsing
            continue

# Separate features (X) and labels (y)
X = []
y = []
for (temp, humid, rain_val) in data:
    X.append([temp, humid])
    # Convert 1 -> +1 (Rain), 0 -> -1 (No Rain)
    if rain_val == 1:
        y.append(1)
    else:
        y.append(-1)

# Split into training (first 15) and testing (last 5)
train_data = X[:15]
train_labels = y[:15]
test_data = X[15:]
test_labels = y[15:]


# Print all test/train data
# print("Test Data:")
# for row in train_data:
#     print(row)

# print("\nTest Labels:")
# for label in train_labels:
#     print(label)

perceptron = p(l_rate=0.1, n_iter=1000)
perceptron.fit(train_data, train_labels)

test_preds = perceptron.stepFunction(test_data)
print(test_preds)

# Assuming y_preds and test_labels are lists of predicted and true labels

# Count the number of correct predictions
correct = 0
for pred, true in zip(test_preds, test_labels):
    if pred == true:
        correct += 1

# Compute accuracy as a fraction of correct predictions
accuracy = correct / len(test_labels)

# Print the accuracy as a percentage rounded to two decimal places
print("Accuracy:", round(accuracy * 100, 2), "%")