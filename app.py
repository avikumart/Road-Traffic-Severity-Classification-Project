import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib

model = joblib.load("rta_model_deploy3.joblib")
encoder = joblib.load("ordinal_encoder2.joblib")

# 1: serious injury, 2: Slight injury, 0: Fatal Injury

st.set_page_config(page_title="Accident Severity Prediction App",
                page_icon="🚧", layout="wide")

#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

# number of vehical involved: range of 1 to 7
# number of casualties: range of 1 to 8
# hour of the day: range of 0 to 23

options_types_collision = ['Vehicle with vehicle collision','Collision with roadside objects',
                           'Collision with pedestrians','Rollover','Collision with animals',
                           'Unknown','Collision with roadside-parked vehicles','Fall from vehicles',
                           'Other','With Train']

options_sex = ['Male','Female','Unknown']

options_education_level = ['Junior high school','Elementary school','High school',
                           'Unknown','Above high school','Writing & reading','Illiterate']

options_services_year = ['Unknown','2-5yrs','Above 10yr','5-10yrs','1-2yr','Below 1yr']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

# features list
features = ['Number_of_vehicles_involved','Number_of_casualties','Hour_of_Day','Type_of_collision','Age_band_of_driver','Sex_of_driver',
       'Educational_level','Service_year_of_vehicle','Day_of_week','Area_accident_occured']

# take input 
st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
       with st.form("road_traffic_severity_form"):
              st.subheader("Enter the following inputs:")
              
              No_vehicles = st.slider("Number of vehicles involved:",1,7, value=0, format="%d")
              No_casualties = st.slider("Number of casualities:",1,8, value=0, format="%d")
              Hour = st.slider("Hour of the day:", 0, 23, value=0, format="%d")
              collision = st.selectbox("Type of collision:",options=options_types_collision)
              Age_band = st.selectbox("Driver age group?:", options=options_age)
              Sex = st.selectbox("Sex of the driver:", options=options_sex)
              Education = st.selectbox("Education of driver:",options=options_education_level)
              service_vehicle = st.selectbox("Service year of vehicle:", options=options_services_year)
              Day_week = st.selectbox("Day of the week:", options=options_day)
              Accident_area = st.selectbox("Area of accident:", options=options_acc_area)
              
              submit = st.form_submit_button("Predict")

# encode using ordinal encoder and predict
       if submit:
              input_array = np.array([collision,
                                   Age_band,Sex,Education,service_vehicle,
                                   Day_week,Accident_area], ndmin=2)
              
              encoded_arr = list(encoder.transform(input_array).ravel())
              
              num_arr = [No_vehicles,No_casualties,Hour]
              pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)              
          
              prediction = model.predict(pred_arr)
              
              if prediction == 0:
                     st.write(f"The severity prediction is Fatal Injury")
              elif prediction == 1:
                     st.write(f"The severity prediction is serious injury")
              else:
                     st.write(f"The severity prediciton is slight injury")
                  
if __name__ == '__main__':
   main()
                     
              

       
       



