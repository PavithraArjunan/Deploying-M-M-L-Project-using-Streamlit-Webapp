This project is a **Streamlit-based web application** that predicts the likelihood of three diseases:

- **Diabetes Detection** (SVM Model)
- **Heart Disease Detection** (Logistic Regression Model)
- **Parkinson's Disease Detection** (SVM Model)

The web app allows users to input relevant medical parameters and get predictions based on trained machine learning models.

## Dependencies
The following Python libraries are required to run this project:

```python
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
```

## Project Structure
- `diabetes_model.sav` - Pre-trained model for Diabetes Detection
- `heartdisease_model.sav` - Pre-trained model for Heart Disease Detection
- `parkinsons_model.sav` - Pre-trained model for Parkinson’s Disease Detection
- `app.py` - Streamlit web application file

## How to Run the Application
1. Install the required dependencies:
   ```bash
   pip install numpy pickle-mixin streamlit streamlit-option-menu
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. The application will open in your default browser, allowing you to choose a disease prediction model and input necessary medical values.

## Implementation Details
### 1. Loading Trained Models
```python
diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))
heart_model = pickle.load(open("heartdisease_model.sav", 'rb'))
parkinson_model = pickle.load(open("parkinsons_model.sav", 'rb'))
```

### 2. Sidebar Navigation
```python
with st.sidebar:
    selected = option_menu('DISEASE PREDICTION',
                           ['DIABETES DETECTION PAGE',
                           'HEART DISEASE DETECTION PAGE',
                           'PARKINSON DISEASE DETECTION PAGE'],
                           icons=['activity','heart','person-circle'],
                           default_index=0)
```

### 3. Prediction Functions
#### Diabetes Prediction
- Users input 8 medical parameters.
- The model predicts whether the person has diabetes or not.

```python
if st.button('Diabetes Test Result'):
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    result = 'Diabetic' if diab_prediction[0] == 1 else 'Not Diabetic'
    st.success(result)
```

#### Heart Disease Prediction
- Users input 13 medical parameters.
- Logistic Regression model predicts heart disease risk.

```python
def heart_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.number).reshape(1, -1)
    prediction = heart_model.predict(input_data)
    return 'Heart Patient' if prediction[0] == 1 else 'Healthy'
```

#### Parkinson’s Disease Prediction
- Users input 22 medical parameters.
- The model predicts the likelihood of Parkinson’s disease.

```python
if st.button("Parkinson's Test Result"):
    parkinsons_prediction = parkinson_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
    result = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
    st.success(result)
```

## Conclusion
This web application provides an interactive platform for disease prediction using pre-trained ML models. It allows users to input relevant parameters and receive instant predictions. **Streamlit** makes it easy to use, and the models provide quick and efficient results.

---


