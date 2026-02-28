# Churn Prediction using Machine Learning by Ankit Singh


# Live Link -  https://churn-prediction-2-4prs.onrender.com 



##             Problem Type

Supervised Binary Classification




##                 Why This Dataset is Good for ML

 Real-world business problem
 Mix of categorical and numerical features
 Balanced enough for classification
 Allows feature engineering



##                  Dataset Size (Based on Your Notebook)

From your .info() output:
Number of rows → ~1000 customers
Number of columns → ~7–8 features




#              About the Project - 
This is a Machine Learning project where I built a Customer Churn
Prediction System** using Python and Streamlit.

The goal of this project is to predict whether a customer will leave
(churn) or stay based on: - Age - Gender - Tenure - Monthly Charges

This project helped me understand: - Data preprocessing - Feature
scaling - Model training - Model deployment using Streamlit



##              What i Did - 

This is an end-to-end customer churn prediction project. I started with data cleaning and EDA to understand customer behavior patterns. I selected relevant billing and demographic features, encoded categorical variables, and applied standard scaling. I trained multiple ML models including Logistic Regression, KNN, SVM, Decision Tree, and Random Forest with hyperparameter tuning using GridSearchCV. Finally, I selected the best-performing model and saved it for deployment.


##               Problem Statement

Customer churn is a major issue for companies.
The objective of this project is to build a model that predicts:

-   1 → Customer will churn
-   0 → Customer will not churn

##               Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Streamlit
-   Matplotlib




##                Project Structure

    Customer-Churn-Prediction/
    │
    ├── train_model.ipynb      # Model training notebook
    ├── app.py                 # Streamlit web application
    ├── model.pkl              # Saved trained model
    ├── customer_churn_data.csv
    └── README.md



##                 Machine Learning Workflow

1. Data Loading

The dataset was loaded using Pandas.

2.  Feature Selection

Input features (X): - Age - Gender - Tenure - MonthlyCharges

Target variable (y): - Churn

3. Data Preprocessing

-   Converted Gender:
    -   Male → 0
    -   Female → 1
-   Split dataset into Training and Testing sets.

4.  Feature Scaling

Used **StandardScaler** to normalize the data because: - Features had
different ranges. - Scaling improves model performance.

The scaler was saved as `scaler.pkl`.

5.   Model Training

Used **Logistic Regression** for binary classification.

Why Logistic Regression? - Suitable for 0/1 prediction. - Simple and
interpretable. - Works well for structured tabular data.

The trained model was saved as `model.pkl`.

6.   Model Evaluation

Model performance was evaluated using accuracy score on test data.







##             Streamlit Web Application

The app takes user inputs: - Age - Gender - Tenure - Monthly Charges

Steps inside the app: 1. Load saved model and scaler. 2. Convert Gender
into numeric format. 3. Scale input using saved scaler. 4. Predict churn
using trained model. 5. Display prediction result.

To run the app:

"Terminal"
run - streamlit run app.py
```




##                  Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Streamlit
-   Joblib





##                   What I Learned

-   End-to-end Machine Learning workflow
-   Importance of scaling in ML
-   Difference between training and testing data
-   Saving and loading models
-   Deploying ML models using Streamlit





##                   Future Improvements

-   Add Exploratory Data Analysis (EDA)
-   Add Confusion Matrix and ROC Curve
-   Try advanced models (Random Forest, XGBoost)
-   Improve UI design of Streamlit app






##                    Conclusion

This project demonstrates my understanding of: - Machine Learning
- Data preprocessing - Model building - Deployment of ML models
, this project helped me build a strong foundation in
applied Machine Learning.
Thanks for checking my project! 
- Ankit Singh



##                    Steps Performed    

1. Importing Required Libraries
2. Loading the Dataset
3. Understanding the Dataset
4. Data Cleaning
5. Exploratory Data Analysis (EDA)
6. Feature Selection
7. Encoding Categorical Variable
8. Train-Test Split, Feature Scaling
9. Feature Scaling, 
10. Model Training, 
11. Hyperparameter Tuning
12. Model Evaluation
13. Model Selection
14. Model Saving




###                   Data Preprocessing

-   Handled missing values
-   Encoded categorical variables
-   Feature scaling




###                    Model Training

-   Split data into train and test sets
-   Trained classification model (e.g., Logistic Regression / Random
    Forest)
-   Evaluated using accuracy score







##                  How to Run the Project

### Step 1: Clone the repository

    git clone <your-repository-link>
    cd Customer-Churn-Prediction

### Step 2: Create virtual environment (optional but recommended)

    python -m venv venv
    venv\Scripts\activate

### Step 3: Install dependencies

    pip install -r requirements.txt

### Step 4: Run Streamlit app

    streamlit run app.py








##                     Steps Performed

### 1️. Data Preprocessing

-   Handled missing values
-   Encoded categorical variables
-   Feature scaling

### 2️. Model Training

-   Split data into train and test sets
-   Trained classification model (e.g., Logistic Regression / Random
    Forest)
-   Evaluated using accuracy score

### 3️. Model Deployment

-   Saved trained model using pickle
-   Built interactive UI using Streamlit
-   Deployed application for real-time predictions






##                   Model Output

The application takes user input and predicts: -  Customer will stay -
 Customer will churn







##                    Dataset Information - 

Dataset source (e.g., Kaggle)
Number of rows
Number of features
Target variable







##                     Model Pipelines

1. Data Cleaning
2. Handling Missing Values
3. Encoding Categorical Variables
4. Feature Scaling
5. Train-Test Split
6. Model Training (Logistic Regression / Random Forest)
7. Model Evaluation
8. Model Deployment using Streamlit






#                      Model Used, Type,  Why?

1. Logistic Regression, Linear, Baseline model
2. K-Nearest Neighbors (KNN), Distance-based, Non-linear comparison
3. Support Vector Machine (SVM), Margin-based, Strong classifier
4. Decision Tree, Tree-based, Interpretability
5. Random Forest,  Ensemble, Improved performance






#                      Model Performance
Logistic Regression - 90.50 %
K-Nearest Neighbors(KNN) - 90.00 %
SVM - 90.00 %
Decision Tree - 90.00
Random Forest - 90.00 %





##                    .gitignore 

1. .env 






