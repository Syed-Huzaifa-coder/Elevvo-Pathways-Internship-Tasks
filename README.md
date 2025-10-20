# Elevvo-Pathways-Internship-Tasks
Remote Machine Learning Internship Opportunity in Cairo, Egypt.

This repository contains all project tasks completed during my Machine Learning Internship at Elevvo Pathways, Cairo, Egypt.
Each task demonstrates practical applications of core ML techniques — from regression and clustering to classification and recommendation systems.

📋 Overview
| Level | Task   | Topic                            | Main Techniques                                 |
| :---: | ------ | -------------------------------- | ----------------------------------------------- |
|  1️⃣  | Task 1 | Student Exam Score Prediction    | Linear Regression                               |
|  1️⃣  | Task 2 | Customer Segmentation            | K-Means, DBSCAN                                 |
|  2️⃣  | Task 3 | Forest Cover Type Classification | Random Forest, XGBoost                          |
|  2️⃣  | Task 4 | Loan Approval Prediction         | Logistic Regression, Decision Tree, SMOTE       |
|  2️⃣  | Task 5 | Movie Recommendation System      | User-Based & Item-Based Collaborative Filtering |

🧩 Level 1 Tasks
🧮 Task 1: Student Exam Score Prediction

Goal:
Predict student exam scores based on the number of study hours.

Dataset: Custom synthetic dataset (student_scores.csv)

Steps:
Load and explore dataset
Visualize relationship between hours_studied and exam_score
Train a Linear Regression model
Evaluate using RMSE and R² score

Tools & Libraries:
Python, Pandas, Matplotlib, Seaborn, Scikit-learn

Bonus: Visualized regression line and residuals.

🛍️ Task 2: Customer Segmentation (Clustering)

Goal:
Group mall customers into clusters based on annual income and spending score.

Dataset: Mall Customers Dataset (Kaggle)

Steps:
Data preprocessing and scaling
Visualize customer distributions
Determine optimal clusters using Elbow Method
Apply K-Means Clustering
Visualize cluster boundaries in 2D

Bonus:
Compared DBSCAN clustering results
Analyzed average spending per cluster

Tools & Libraries:
Python, Pandas, Matplotlib, Seaborn, Scikit-learn

⚙️ Level 2 Tasks
🌲 Task 3: Forest Cover Type Classification

Goal:
Predict the type of forest cover using cartographic and environmental features.

Dataset: Covertype Dataset (UCI)

Steps:
Data cleaning and feature scaling
Encoded categorical variables

Trained multiple models:
Random Forest
XGBoost

Evaluated performance using accuracy, balanced accuracy, and classification report
Visualized confusion matrix and feature importance

Bonus:
Hyperparameter tuning using RandomizedSearchCV
Compared RandomForest vs XGBoost performance

Tools & Libraries:
Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn

💰 Task 4: Loan Approval Prediction

Goal:
Predict whether a loan application will be approved.

Dataset: Loan Approval Prediction Dataset (Kaggle)

Steps:
Handle missing values
Encode categorical features
Split into train-test sets

Train classification models:
Logistic Regression
Decision Tree

Evaluate using:
Precision
Recall
F1-Score

Bonus:
Applied SMOTE to handle class imbalance
Compared model performance on imbalanced vs. balanced data

Tools & Libraries:
Python, Pandas, Scikit-learn, Imbalanced-learn

🎬 Task 5: Movie Recommendation System

Goal:
Recommend movies to users based on historical rating patterns.

Dataset: MovieLens 100K Dataset (Kaggle)

Steps:
Create user–item rating matrix
Compute user similarity using cosine similarity
Recommend top-rated unseen movies for each user
Evaluate using Precision@K metric

Bonus:
Implemented item-based collaborative filtering
Used SVD (Matrix Factorization) for performance comparison

Tools & Libraries:
Python, Pandas, NumPy, Scikit-learn, Surpris

🧰 General Tools Used
Programming Language: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn
Environment: Jupyter Notebook / Google Colab

🧑‍💻 About the Internship
Organization: Elevvo Pathways
Location: Cairo, Egypt
Role: Machine Learning Intern
Focus Areas:
Regression, Clustering, Classification, Recommendation Systems, Model Evaluation & Optimization

🏁 Results Summary:
| Task | Model(s) Used               | Best Performance Metric            |
| ---- | --------------------------- | ---------------------------------- |
| 1    | Linear Regression           | R² ≈ 0.95                          |
| 2    | K-Means (k=5)               | Clear customer clusters visualized |
| 3    | XGBoost                     | Balanced Accuracy ≈ 0.88           |
| 4    | Logistic Regression (SMOTE) | F1 ≈ 0.82                          |
| 5    | User-Based CF               | Precision@K ≈ 0.78                 |

📜 Author

Name: Syed Huzaifa Bin Khamis
Role: Machine Learning Intern @ Elevvo Pathways
Location: Cairo, Egypt
LinkedIn: linkedin.com/in/syedhuzaifabinkhamis



