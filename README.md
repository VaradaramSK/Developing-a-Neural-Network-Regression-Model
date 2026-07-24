# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.
## Neural Network Model
<img width="1116" height="757" alt="image" src="https://github.com/user-attachments/assets/bf1e1aad-fecb-4ecb-a3cf-1e1f04db0f82" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Varadaram SK

### Register Number: 212223040232

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = df = pd.read_csv(r"C:\Users\admin\Desktop\Deep learning\Exp-1.csv")    # Change the path if needed

# Input and Output
x = df[["Input"]].values
y = df[["Output"]].values

# Split data
xt, xst, yt, yst = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Scale data
scale1 = MinMaxScaler()
scale2 = MinMaxScaler()

xt = scale1.fit_transform(xt)
xst = scale1.transform(xst)

yt = scale2.fit_transform(yt)
yst = scale2.transform(yst)

# Convert to tensors
xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)

# Neural Network
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    prediction = model(xt)
    loss = criterion(prediction, yt)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss = {loss.item():.6f}")
        losses.append(loss.item())

# Predict new value
new = scale1.transform([[16]])
new = torch.FloatTensor(new)

with torch.no_grad():
    pred = model(new)
    pred = scale2.inverse_transform(pred.numpy())
    print("Predicted Output:", pred[0][0])

# Test Loss
with torch.no_grad():
    pred_test = model(xst)
    test_loss = criterion(pred_test, yst)
    print("Test Loss:", test_loss.item())

# Plot loss
plt.plot(range(0, epochs, 50), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
```

### Dataset Information
<img width="432" height="414" alt="image" src="https://github.com/user-attachments/assets/80fa360f-e2c8-4992-98b8-516feba778f7" />



### OUTPUT
<img width="425" height="440" alt="image" src="https://github.com/user-attachments/assets/40f97bbd-2cbf-4ea8-99ba-bb18aca2f020" />


### Training Loss Vs Iteration Plot
<img width="787" height="596" alt="image" src="https://github.com/user-attachments/assets/b0a67747-8f63-428b-aceb-307880ca8034" />



### New Sample Data Prediction
<img width="755" height="44" alt="image" src="https://github.com/user-attachments/assets/f800c4a3-af20-4ca7-8ece-853897ad4e89" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
