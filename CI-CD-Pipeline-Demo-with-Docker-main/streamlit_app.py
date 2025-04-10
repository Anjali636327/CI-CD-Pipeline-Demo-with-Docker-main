import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('Cardetails.csv')

# Preprocessing mileage column
data['mileage'] = data['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '', regex=False)
data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')
data['mileage'] = data['mileage'].fillna(data['mileage'].mean())

# Preprocessing engine column
data['engine'] = data['engine'].str.replace(' CC', '', regex=False)
data['engine'] = pd.to_numeric(data['engine'], errors='coerce')
data['engine'] = data['engine'].fillna(data['engine'].mean())

# Preprocessing max_power column
data['max_power'] = data['max_power'].str.replace(' bhp', '', regex=False)
data['max_power'] = pd.to_numeric(data['max_power'], errors='coerce')
data['max_power'] = data['max_power'].fillna(data['max_power'].mean())

# Preprocessing seats column
data['seats'] = data['seats'].fillna(data['seats'].mode()[0])  # Replace NaN with mode

# Splitting features and target
X = data.drop(columns=['selling_price'], axis=1)
y = data['selling_price']

# Handling categorical columns
categorical_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
numerical_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Model pipeline with optimized RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))  # Reduced complexity
])

# Train the model (only once)
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    # Save the trained model to disk
    joblib.dump(model, 'car_price_model.pkl')

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('car_price_model.pkl')

# Load model only if it's not already cached
try:
    model = load_model()
except:
    train_model()
    model = load_model()

# Combined styling (previous styles + green sliders)
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #6a11cb, #2575fc); /* Gradient background */
        font-family: 'Arial', sans-serif;
        color: black;
        margin: 0;
        padding: 0;
        background-color:green;
    }
    .stButton>button {
        background-color: purple;
        color: black;
        font-size: 20px;
        font-weight: bold;
        padding: 15px 40px;
        border-radius: 90px;
        border: none;
        cursor: pointer;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #599650;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        border: 3.25px solid #920e9e;
    }
    .stSelectbox>div, .stSlider>div, .stNumberInput>div {
        color: blue;
        background-color: white;
    }
    h1 {
        background: #a854b0; /* Gradient text color */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stAlert {
        background-color: white;
        color: black; /* Black text for output */
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        padding: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    /* Updated Green Sliders with Darker Track */
    .stSlider > div > div > div > div {
        background: green !important; /* Slider thumb handle */
        border: none;
    }
    .stSlider > div > div > div {
        #228B22 !important; /* Darker green for the slider track */
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("CAR PRICE PREDICTION")

# Input fields for prediction
name = st.selectbox("Car Name", X['name'].unique())
year = st.slider("Year of Manufacture", int(X['year'].min()), 2025, 2015)  # Updated to 2025
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel = st.selectbox("Fuel Type", X['fuel'].unique())
seller_type = st.selectbox("Seller Type", X['seller_type'].unique())
transmission = st.selectbox("Transmission Type", X['transmission'].unique())
owner = st.selectbox("Owner Type", X['owner'].unique())
mileage = st.number_input("Mileage (kmpl or km/kg)", min_value=0.0, value=15.0)
engine = st.number_input("Engine Capacity (CC)", min_value=0.0, value=1200.0)
max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=75.0)
seats = st.slider("Number of Seats", 2, 10, 5)  # Green slider for seats

# Prediction button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    # Predicting price (cached function to avoid re-processing)
    @st.cache_data
    def predict_price(input_data):
        processed_data = model.named_steps['preprocessor'].transform(input_data)  # Ensure the preprocessor is fitted
        prediction = model.named_steps['model'].predict(processed_data)
        return prediction

    prediction = predict_price(input_data)
    st.success(f"Estimated Selling Price: ₹{prediction[0]:,.2f}")