# RNN_Stock_Price_Prediction
The objective of this assignment is to try and predict the stock prices using historical data from four companies: IBM (IBM), Google (GOOGL), Amazon (AMZN), and Microsoft (MSFT). We use four different companies because they belong to the same sector: Technology. Using data from all four companies may improve the performance of the model.
---

## Dataset

Stock data for:
- Amazon (AMZN)
- Microsoft (MSFT)
- Google (GOOGL)
- IBM

Each CSV file contains historical daily stock information including `Open`, `High`, `Low`, `Close`, `Volume`, and `Adj Close`.

---

## Tools & Libraries

- **Python 3.8+**
- **Pandas**, **NumPy** – data manipulation
- **Matplotlib**, **Seaborn** – data visualization
- **scikit-learn** – scaling and preprocessing
- **TensorFlow / Keras** – building and training RNN model

---
### 1. Data Preparation
- Loaded and cleaned 4 stock CSVs
- Combined into a master DataFrame
- Extracted `Close` prices for target prediction
- Windowed dataset:  
  - `window_size = 65`  
  - `stride = 5`  
  - `test_size = 20%`

### 2. Exploratory Data Analysis
- Frequency distributions of volumes
- Volume variation over time
- Correlation between stock features

### 3. Data Preprocessing
- MinMax Scaling applied to all features
- Time-series windowing for `X` and multi-output `y`

### 4. Model Architecture
Custom RNN model built using `Keras`:
- Multiple RNN layers
- Dropout regularization
- Output layer with 4 neurons for 4 stock targets

### 5. Hyperparameter Tuning
Performed **Manual Grid Search**:
- `128` RNN units  
- `2` layers  
- `0.3` dropout  
- `rmsprop` optimizer with `0.001` learning rate  
- `batch size` of 32, `10` epochs`

### 6. Final Model Results
```
Final Test Loss (MSE): 46493.808594
Final Test MAE: 210.395966
