# Air Quality Index (AQI) Prediction 🌬️

## Project Overview
This machine learning project predicts Air Quality Index values by analyzing various meteorological factors. By understanding the relationship between weather conditions and air quality, we can help forecast potential pollution levels and their impact on public health.

## Dataset
The analysis uses 'AQI_Data.csv' containing meteorological parameters and their corresponding AQI values including:
- Temperature measurements (T, TM, Tm)
- Sea level pressure (SLP)
- Humidity (H)
- Wind-related measurements (W, V, VM)
- AQI values (target variable)

## Implementation Steps

### 1. Data Preprocessing 🔍
```python
# Check for null values
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')

# Drop rows with null values
df = df.dropna()

# Split data into features and target
X = df.iloc[:,:-1]  # independent features
Y = df.iloc[:,-1]   # dependent feature (AQI)
```

- Handled missing values through visualization and removal
- Separated features from target variable for model training

### 2. Exploratory Data Analysis
```python
# Generate correlation matrix
df.corr()  # negative correlation with T,TM,Tm, W,V,VM and positive correlation with SLP,H

# Create detailed correlation heatmap
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
```

The EDA revealed interesting patterns in the data:
- Negative correlations between AQI and temperature variables (T, TM, Tm)
- Negative correlations with wind measurements (W, V, VM)
- Positive correlations with sea level pressure (SLP) and humidity (H)

### 3. Feature Importance Analysis
```python
# Identify most influential features using ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X, Y)

# Visualize feature importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
```

This analysis revealed which meteorological factors have the strongest influence on air quality, guiding our understanding of pollution dynamics.

### 4. Model Building and Evaluation
```python
# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Train Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Evaluate model performance
print("R² on train set:", regressor.score(X_train, Y_train))
print("R² on test set:", regressor.score(X_test, Y_test))

# Cross-validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor, X, Y, cv=5)
print("Cross-validation score:", score.mean())
```

### 5. Model Analysis and Predictions
```python
# Get model coefficients
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])

# Make predictions
prediction = regressor.predict(X_test)

# Visualize results
sns.distplot(Y_test - prediction)  # Error distribution
plt.scatter(Y_test, prediction)    # Actual vs Predicted

# Calculate error metrics
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, prediction))
print('MSE:', metrics.mean_squared_error(Y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, prediction)))
```

## Key Findings
The model demonstrates how different weather parameters influence air quality, with particular attention to temperature and wind patterns. The prediction accuracy provides valuable insights for environmental monitoring and public health planning.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation and Usage
1. Clone the repository
2. Install required dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Ensure 'AQI_Data.csv' is in the same directory
4. Run the Jupyter notebook cells sequentially

## Model Performance
The Linear Regression model serves as a baseline approach with:
- R² scores on training and test sets
- 5-fold cross-validation results
- Comprehensive error analysis (MAE, MSE, RMSE)
- Visualization of prediction accuracy

## Future Improvements
- Implement more advanced regression algorithms (Random Forest, Gradient Boosting)
- Incorporate feature engineering to enhance predictive power
- Explore time-series aspects of air quality fluctuations
- Add geospatial analysis to account for regional variations
- Hyperparameter tuning for optimal performance

## Learning Outcomes 📚
This project demonstrates:
- Data preprocessing and cleaning techniques
- Exploratory data analysis with correlation analysis
- Feature importance evaluation
- Machine learning model implementation and evaluation
- Error analysis and visualization techniques
- Cross-validation for model reliability assessment

---

*This project serves as both a practical implementation of machine learning for environmental data and an educational resource for those interested in applying data science to climate and
