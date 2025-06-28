# 🏥 Injury Risk Prediction using XGBoost and Streamlit

This project is a machine learning-based application that predicts the likelihood of injury in athletes based on physical attributes and training data. It uses an XGBoost classifier for prediction and is deployed as a user-friendly web app using Streamlit.

---

## 📌 Objective

The goal of this project is to identify high-risk injury scenarios in athletes based on historical data and training metrics. By inputting details such as age, weight, height, training intensity, recovery time, and previous injuries, the model predicts whether an athlete is likely to sustain an injury.

---

## 🧠 Machine Learning Workflow

### 🔹 Dataset
The dataset contains 1,000 samples with the following features:
- `Player_Age`
- `Player_Weight`
- `Player_Height`
- `Previous_Injuries` (0 or 1)
- `Training_Intensity` (scaled between 0.0–1.0)
- `Recovery_Time` (days)
- `Likelihood_of_Injury` (target variable: 0 = No Injury, 1 = Injury)

### 🔹 Preprocessing
- Handled missing values (if any)
- Normalized features using `StandardScaler`
- Split data into training and testing sets (80:20)

### 🔹 Model
- Used **XGBoost Classifier** with `eval_metric='logloss'`
- Compared performance against Logistic Regression and SVM
- Evaluated using accuracy, precision, recall, F1-score, and confusion matrix
- Visualized feature importance to understand key risk factors

---

## ⚙️ Tools & Libraries

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib / Seaborn (for visualization)

---

## 📈 Results

- XGBoost achieved ~77% accuracy on the test set (baseline performance)
- Model performance could be impacted by class imbalance or limited features
- Important features: **Previous_Injuries**, **Recovery_Time**, **Training_Intensity**

---

## 🔮 Future Improvements

- ✅ Add more features like player position, training schedule, match load
- ✅ Tune hyperparameters of the XGBoost model (GridSearchCV or Optuna)
- ✅ Use SHAP or LIME for explainability
- ✅ Deploy the app online using Streamlit Cloud or Heroku
- ✅ Support CSV file uploads for batch predictions


## 🚀 Streamlit App

An interactive **Streamlit** dashboard allows users to:
- Input athlete data through sliders and dropdowns
- Receive instant injury risk predictions
- View probability score of the prediction

To run the app locally:
```bash
streamlit run injury_dashboard.py
