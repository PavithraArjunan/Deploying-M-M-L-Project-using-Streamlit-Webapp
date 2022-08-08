import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Loading the saved model

diabetes_model = pickle.load(open("C:/Users/Lenovo/Documents/ML/diadetes_model.sav",'rb'))
heart_model = pickle.load(open("C:/Users/Lenovo/Documents/ML/heartdisease_model.sav",'rb'))
parkinson_model = pickle.load(open("C:/Users/Lenovo/Documents/ML/parkinsons_model.sav",'rb'))

# sidebar for navigation

with st.sidebar:
    selected = option_menu('DISEASE PREDICTION',
                           ['DIABETES DETECTION PAGE',
                           'HEART DISEASE DETECTION PAGE',
                            'PARKINSON DISEASE DETECTION PAGE'],
                           icons =['activity','heart','person-circle'],
                           # icons aretaken from getbootstrap
                           default_index=0)
    
# Diabetes prediction page
if(selected =='DIABETES DETECTION PAGE'):
    
    # Page Title
    st.title('DIABETES DETECTION ')
    
    col1, col2, col3 = st.columns(3)
    
    with col1: 
        Pregnancies = st.text_input("Pregnancies")
    
    with col2: 
        Glucose = st.text_input("Glucose level")
    
    with col3: 
        BloodPressure = st.text_input("Blood Pressure Level")
    
    with col1: 
        SkinThickness = st.text_input("Skin Thickness Level")
    
    with col2: 
        Insulin = st.text_input("Insulin Level")
    
    with col3: 
        BMI = st.text_input("Body Mass Index Value")
    
    with col1: 
        DiabetesPedigreeFunction = st.text_input("DPF")
    
    with col2: 
        Age = st.text_input("Age of the Patient")
        
        
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)



# Heard prediction page
if(selected =='HEART DISEASE DETECTION PAGE'):
    
   
    def heart_prediction(input_data):
        
        input_data_as_numpy_array = np.asarray(input_data,dtype=np.number)
        input_data = input_data_as_numpy_array.reshape(1,-1)
        prediction = heart_model.predict(input_data)
        print(prediction)
        
        if (prediction [0] == 1):
            return 'HEART PATIENT'
        else:
            return 'HEALTHY'


    def main():
        
        # Giving the title
        st.title('HEART  DISEASE  DETECTION ')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.text_input('Age')
            
        with col2:
            sex = st.text_input('Sex')
            
        with col3:
            cp = st.text_input('Chest Pain types')
            
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
            
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
            
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
            
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
            
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
            
        with col3:
            exang = st.text_input('Exercise Induced Angina')
            
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
            
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
            
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')
            
        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
       
        # Code for Prediction
        diagnosis = ''
        
        # Creating the button 
        if st.button('Test Result'):
            diagnosis = heart_prediction([age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal])
        
        st.success(diagnosis)
        
    if __name__ == '__main__':
        main()
        
        
        


# Parkinson prediction page
if(selected =='PARKINSON DISEASE DETECTION PAGE'):
    
    # Page Title
    st.title('PARKINSON  DISEASE  DETECTION ')
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinson_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson disease"
        
    st.success(parkinsons_diagnosis)



