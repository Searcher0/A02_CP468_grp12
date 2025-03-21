import csv
from Class import Perceptron as p

# Read CSV file and store the data
data = []
with open('WeatherData_Q3.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        try:
            # parse columns as floats
            temp = float(row[0])
            humid = float(row[1])
            label = int(row[2])
            data.append((temp, humid, label))
        except ValueError:
            continue

# Separate features (X) and labels (y)
X = []
y = []
for (temp, humid, rain_val) in data:
    X.append([temp, humid])
    # Converting [1 into +1 (Rain)] & [0 into -1 (No Rain)]
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

# Count the number of correct predictions
correct = 0
for pred, true in zip(test_preds, test_labels):
    if pred == true:
        correct += 1

# Compute accuracy
accuracy = correct / len(test_labels)
print("Accuracy:", round(accuracy * 100, 2), "%")