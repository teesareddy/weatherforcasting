import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the Excel file
file_path = 'sample_weather_data.xlsx'  # Update this path to your actual file
df = pd.read_excel(file_path)

# Preprocess the data
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour

# Drop unnecessary columns
df.drop(['date'], axis=1, inplace=True)

# Define the feature and target variables
X = df.drop(['temperature', 'weather'], axis=1)
y = df['temperature']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Train a random forest regressor model
#The RandomForestRegressor from the sklearn.ensemble library is used to create the regressor model. 
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Use the model to make predictions on new data
new_data = pd.DataFrame({
    'month': [7, 8, 9, 10, 11], 
    'day': [1, 15, 30, 1, 15], 
    'hour': [12, 12, 12, 12, 12], 
    'humidity': [60, 60, 60, 60, 60], 
    'wind_speed': [10, 10, 10, 10, 10]
})

new_data['month'] = new_data['month'].astype(str)
new_data['day'] = new_data['day'].astype(str)
new_data['hour'] = new_data['hour'].astype(str)
new_data = pd.get_dummies(new_data, columns=['month', 'day', 'hour'])
new_data = new_data[['month_7', 'day_1', 'hour_12', 'humidity', 'wind_speed']]

# Make predictions on the new data
predictions = model.predict(new_data)

# Create a new DataFrame for the weather conditions
weather_conditions = pd.DataFrame({
    'month': [7, 8, 9, 10, 11], 
    'day': [1, 15, 30, 1, 15], 
    'hour': [12, 12, 12, 12, 12], 
    'humidity': [60, 60, 60, 60, 60], 
    'wind_speed': [10, 10, 10, 10, 10], 
    'weather': ['rain', 'sunny', 'windy', 'rain', 'sunny']
})

# Make predictions on the weather conditions
weather_predictions = model.predict(weather_conditions)

# Write the predictions to a new Excel sheet
weather_conditions['prediction'] = weather_predictions
weather_conditions.to_excel('weather_predictions.xlsx', index=False)