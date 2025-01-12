import numpy as np
#the given data
data = {
    "Bug ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Title": [
        "UI Button Misaligned", "Crash on Login", "Typo in User Guide", 
        "Payment Gateway Down", "Search Bug", "Profile Picture Error", 
        "Email Not Sent", "Slow Dashboard Load", "Language Selector Bug", 
        "Duplicate Records"
    ],
    "Description": [
        "Button overlaps with text", "App crashes on invalid input", "Minor typo in documentation",
        "Payments fail intermittently", "No results for valid queries", 
        "Unable to upload pictures", "Password reset email not sent", 
        "Dashboard takes 10s to load", "Wrong language selected", 
        "Duplicate user records in DB"
    ],
    "Severity": ["High", "Critical", "Low", "Critical", "High", "Medium", "High", "Medium", "Low", "Critical"],
    "Priority": ["P2", "P1", "P3", "P1", "P2", "P2", "P1", "P2", "P3", "P1"],
    "Component": [
        "UI/Frontend", "Authentication", "Documentation", 
        "Payments", "Search Module", "Profile Feature", 
        "Notifications", "Performance", "Localization", 
        "Database"
    ],
    "Assigned Team": ["Frontend", "Backend", "Docs Team", "Payments", "Backend", "Frontend", "Backend", "Backend", "Frontend", "Backend"]
}
# numerical mappings for the categorical data
severity_mapping = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
priority_mapping = {"P1": 3, "P2": 2, "P3": 1}
component_mapping = {
    "UI/Frontend": 0, "Authentication": 1, "Documentation": 2,
    "Payments": 3, "Search Module": 4, "Profile Feature": 5,
    "Notifications": 6, "Performance": 7, "Localization": 8, "Database": 9
}
team_mapping = {"Frontend": 0, "Backend": 1, "Docs Team": 2, "Payments": 3}
#converting into numpy array
X = []
y = []
#processing the data
for i in range(len(data["Bug ID"])):
    title_length = len(data["Title"][i])  
    description_length = len(data["Description"][i])  
    severity = severity_mapping[data["Severity"][i]]
    priority = priority_mapping[data["Priority"][i]]
    component = component_mapping[data["Component"][i]]
    assigned_team = team_mapping[data["Assigned Team"][i]]
#combine all features into a single list
    X.append([title_length, description_length, severity, priority, component])
    y.append(assigned_team)

X = np.array(X)
y = np.array(y)
#knn classification
def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            distance = np.sqrt(np.sum((test_point - train_point) ** 2))
            distances.append((distance, y_train[i]))
        #sort by distance and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(k)]
        #predict the class
        prediction = max(set(neighbors), key=neighbors.count)
        predictions.append(prediction)
    
    return predictions
# spliting the data into training and testimg sets
train_size = int(0.5 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
#make prediction
y_pred = knn_predict(X_train, y_train, X_test, k=3)
#evaluate the model
correct = sum([1 for i in range(len(y_test)) if y_test[i] == y_pred[i]])
accuracy = correct / len(y_test) * 100
#results
print("Predicted:", y_pred)
print("Actual:   ", y_test.tolist())
print(f"Accuracy: {accuracy:.2f}%")
###thank you###