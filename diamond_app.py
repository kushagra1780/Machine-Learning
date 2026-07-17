import streamlit as st
import pandas as pd
import joblib

# --------------------
# Load Saved Files
# --------------------

lr_model = joblib.load("SimpleLinear.joblib")
dt_model = joblib.load("DecisionTree.joblib")
encoder = joblib.load("encoder.joblib")
scaler = joblib.load("scaler.joblib")

# --------------------
# Read Dataset (Only for Structure)
# --------------------

df = pd.read_csv("diamonds.csv")

cat_cols =["cut", "color", "clarity"]

# --------------------
# Streamlit UI
# --------------------

st.set_page_config(page_title="Diamond Price Prediction")

st.title("Diamond Price Prediction")

model = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Decision Tree Regression"]
)

st.subheader("Enter Diamond Details")

carat = st.number_input("Carat", min_value=0.1, value=1.0)

cut = st.selectbox(
    "Cut",
    sorted(df["cut"].unique())
)

color = st.selectbox(
    "Color",
    sorted(df["color"].unique())
)

clarity = st.selectbox(
    "Clarity",
    sorted(df["clarity"].unique())
)

depth = st.number_input("Depth", value=61.5)

table= st.number_input("Table", value=57.0)

x = st.number_input("Length (x)", value=5.5)

y = st.number_input("Width (y)", value=5.5)

z = st.number_input("Height (z)", value=3.5)

# --------------------
# Prediction
# --------------------

if st.button("Predict Price"):

    user = pd.DataFrame({
        "carat":[carat],
        "cut":[cut],
        "color":[color],
        "clarity":[clarity],
        "depth":[depth],
        "table":[table],
        "x":[x],
        "y":[y],
        "z":[z]

    })

    # Encode
    encoded = encoder.transform(user[cat_cols])

    encoded = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # Remove categorical columns
    user = user.drop(columns=cat_cols)

    #Merge
    user = pd.concat([user, encoded], axis =1)

    # Scale
    user = scaler.transform(user)

    #Prediction

    if model == "Linear Regression":

        prediction = lr_model.predict(user)

    else:
        
        prediction = dt_model.predict(user)

    st.success(f"Predicted Diamond Price : ${prediction[0]:.2f}")