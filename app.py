import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import shap
import matplotlib
from IPython import get_ipython
from PIL import Image

model = joblib.load("rta_model_deploy3.joblib")
encoder = joblib.load("ordinal_encoder2.joblib")

st.set_option('deprecation.showPyplotGlobalUse', False)

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
              st.subheader("Pleas enter the following inputs:")
              
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
                     st.write(f"The severity prediction is Fatal Injury⚠")
              elif prediction == 1:
                     st.write(f"The severity prediction is serious injury")
              else:
                     st.write(f"The severity prediciton is slight injury")
                  
              st.subheader("Explainable AI (XAI) to understand predictions")  
              shap.initjs()
              shap_values = shap.TreeExplainer(model).shap_values(pred_arr)
              st.write(f"For prediction {prediction}") 
              shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0],
                              pred_arr, feature_names=features, matplotlib=True,show=False).savefig("pred_force_plot.jpg", bbox_inches='tight')
              img = Image.open("pred_force_plot.jpg")
              st.image(img, caption='Model explanation using shap')
              
              st.write("Developed By: Avi kumar Talaviya")
              st.markdown("""Reach out to me on: [Twitter](https://twitter.com/avikumart_) |
              [Linkedin](https://www.linkedin.com/in/avi-kumar-talaviya-739153147/) |
              [Kaggle](https://www.kaggle.com/avikumart) 
              """)
              

# post the image of the accident

a,b,c = st.columns([0.2,0.6,0.2])
with b:
  st.image("vllkyt19n98psusds8.jpg", use_column_width=True)


# description about the project and code files            

st.subheader("🧾Description:")
st.text("""This data set is collected from Addis Ababa Sub-city police departments for master's research work. 
The data set has been prepared from manual records of road traffic accidents of the year 2017-20. 
All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident.
Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.
""")

st.markdown("Source of the dataset: [Click Here](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)")

st.subheader("🧭 Problem Statement:")
st.text("""The target feature is Accident_severity which is a multi-class variable. 
The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. 
The metric for evaluation will be f1-score
""")

st.markdown("Please find GitHub repository link of project: [Click Here](https://github.com/avikumart/Road-Traffic-Severity-Classification-Project)")                  
                  
if __name__ == '__main__':
   main()
    
   
                
    
                     
              

       
       



