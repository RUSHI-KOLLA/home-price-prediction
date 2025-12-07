# Home Price Prediction

A machine learning project that predicts house prices in India using Random Forest Regression.  This model analyzes various property features to estimate real estate values with high accuracy.

##  Project Overview

This project uses a dataset of Indian housing prices with 23 different features to train a Random Forest Regressor model. The model achieves approximately **97. 7% accuracy** (R² score) in predicting house prices. 

##  Features

The model considers the following features for price prediction:
- Number of bedrooms and bathrooms
- Living area and lot area
- Number of floors
- Waterfront presence
- Number of views
- Condition and grade of the house
- Area of the house (excluding basement)
- Basement area
- Built year and renovation year
- Postal code
- Geographic coordinates (Latitude & Longitude)
- Living area after renovation
- Lot area after renovation
- Number of schools nearby
- Distance from the airport

## 🚀 Getting Started

### Prerequisites

```python
pandas
scikit-learn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RUSHI-KOLLA/home-price-prediction.git
cd home-price-prediction
```

2. Install required packages:
```bash
pip install pandas scikit-learn
```

3.  Ensure you have the `House Price India.csv` dataset in the project directory.

##  Model Performance

The Random Forest Regressor model delivers excellent performance metrics:

- **R² Score**: 0.977 (97.7% accuracy)
- **Mean Absolute Error**: ₹13,896.66
- **Root Mean Squared Error**: ₹51,899.86

##  Usage

Run the Jupyter notebook `house pred (1).ipynb` to:

1. Load and explore the dataset
2. Clean the data by removing null values
3. Split data into training and testing sets (80/20 split)
4. Train the Random Forest Regressor model
5. Make predictions and evaluate performance

### Quick Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('House Price India.csv')
df. dropna(inplace=True)

# Prepare features and target
x = df.drop('Price', axis=1)
y = df['Price']

# Train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Make predictions
y_pred = model. predict(x_test)
```


##  Dataset

The dataset contains **10,051 records** of house sales in India with 23 features including property characteristics, location data, and pricing information.

##  Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning library
  - RandomForestRegressor - Main prediction model
  - Train-test split for data validation
  - Evaluation metrics (R², MAE, RMSE)

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

##  Author

**RUSHI-KOLLA**

##  Contributing

Contributions, issues, and feature requests are welcome!  Feel free to check the issues page. 

##  Show your support

Give a ⭐️ if this project helped you! 

---

**Note**: Make sure to have the `House Price India.csv` dataset in your project directory before running the notebook. 
