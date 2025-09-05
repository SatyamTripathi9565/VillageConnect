# VillageConnect Satisfaction Score Predictor

A Machine Learning project to predict **Satisfaction Score** for rural services (Electricity, Internet, Water, Healthcare) using the **VillageConnect** dataset.  
The model is deployed as an **interactive Streamlit web app** so users can try predictions in real time.

---

## 🚀 Features
- Data cleaning and preprocessing
- Feature engineering with polynomial features
- One-hot encoding for categorical variables
- Model training with **XGBoost Regressor**
- Hyperparameter tuning using **GridSearchCV**
- Model evaluation with **MSE** and **R² Score**
- Interactive prediction via **Streamlit**
- Correlation heatmap visualization

---

## 📂 Project Structure
├── Week9/
│ ├── app.py # Streamlit app
│ ├── project.py # Model training and evaluation
│ ├── villageconnect_large.csv # Dataset
│ ├── village_model_best.pkl # Saved trained model
│ ├── Screenshot.png # Demo screenshot
└── README.md

pip install -r requirements.txt
pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
streamlit

To Run
streamlit run app.py

📌 How to Use the App

Select the Region and Service Type

Enter Availability%, Cost, and Usage Hours

Click Predict to see the satisfaction score

