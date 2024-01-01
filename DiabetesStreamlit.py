import streamlit as st
import pickle

#load model
diabetes_model = pickle.load(open('Classification Diabetes.sav', 'rb'))

#coding streamlit
st.title("Data Mining Prediksi Diabetes dengan Algoritma Decesion Tree")

#input text pada tiap atribut
Age = st.text_input('Input nilai Age')
Gender = st.text_input('Input nilai Gender')
BMI = st.text_input('Input nilai BMI')
HbA1c = st.text_input('Input nilai HbA1c')
FamilyHistoryofDiabetes = st.text_input('Input nilai FamilyHistoryofDiabetes')
Smoking = st.text_input('Input nilai Smoking')
Diet = st.text_input('Input nilai Diet')
Exercise = st.text_input('Input nilai Exercise')
Diagnosis = st.text_input('Input nilai Diagnosis')

#code untuk prediksi
diagnosis = ''
print(diagnosis)
#membuat tombol prediksi
if st.button('Tes Prediksi Diabetes') :
    prediction = diabetes_model.predict([[Age,Gender,BMI,HbA1c,FamilyHistoryofDiabetes,
                                          Smoking,Diet,Exercise,Diagnosis]])

    if(prediction[0] == 1) :
        diagnosis = 'Pasien di Diagnosa Tidak mengidap diabetes'
    else :
        diagnosis = 'Pasien di Diagnosa Mengidap diabetes'
        
    st.success(diagnosis)
