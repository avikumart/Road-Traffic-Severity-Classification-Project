# Road-Traffic-Severity-Classification-Project
This is a multiclass classification project to classify severity of road accidents into three categories. this project is based on real-world data and dataset is also highly imbalanced. Learn more about detailed description and problem with tasks performed.

**Description:** This data set is collected from Addis Ababa Sub-city, Ethiopia police departments for master's research work. The data set has been prepared from manual records of road traffic accidents of the year 2017-20. All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident. Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms. 

**Source of dataset:** [Link to the dataset](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)

**Problem Statement:** The target feature is Accident_severity which is a multi-class variable. The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. Your metric for evaluation will be f1-score.

### Web application

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://road-traffic-severity-prediction.streamlit.app/)

### Web app screenshots:

![image](https://user-images.githubusercontent.com/88608935/227510381-e8d6c1c1-a5ab-4869-8e02-e20a278b64cc.png)

![image](https://user-images.githubusercontent.com/88608935/227510715-348da3f8-fb78-45f8-9b9f-7d2997176d76.png)

![image](https://user-images.githubusercontent.com/88608935/227510607-81f35fda-ec19-4eb1-b886-374f4bbda449.png)



### Published article on Analytics Vidhya

**Link:** [Click here](https://www.analyticsvidhya.com/blog/2023/01/machine-learning-solution-predicting-road-accident-severity/)

### Tasks and techniques used:

**1. Exploratory data analysis**
- Data analysis using `dabl`
- Exploratory data analysis using `matplotlib` and `seaborn`

**2. Data preparation and pre-processing**
- Missing Values Tretment using fillna method
- One Hot encoding using pandas get_dummies
- Feature selection using `chi2` statistic and SelectKBest method
- PCA to reduce dimentinality
- Imbalance data tretment using `SMOTENC` technique

**3. Modelling using sci-kit learn library**
- Baseline model using `RandomForest` using default technique 
- Tuned hyperparameters using `n_estimators` and `max_depth` parameters 

**4. Evaluation**
- Evaluation metric was weighted `f1_score` 
- Baseline model evaluation `fl_score = 61%`
- Final model evaluation `f1_score = 88%`

### Acknowledgement: [TMLC Academy](https://www.themlco.com/Academy/index.html)

### References:

1. [Machine learning mastery](https://machinelearningmastery.com/feature-selection-with-categorical-data/)
2. [Feature engieering techniques](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
