<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Problem Statement:

A Portuguese banking institution has conducted multiple direct marketing campaigns via phone calls to promote term deposit products. The bank aims to improve the efficiency of these campaigns by predicting which clients are likely to subscribe to a term deposit.
The challenge is to analyze historical marketing campaign data and build a predictive machine learning model that accurately classifies whether a client will subscribe ("yes") or not ("no") to a term deposit. This model will be deployed as an interactive web application using Streamlit to assist marketing teams in targeting potential clients more effectively, thereby optimizing campaign performance and reducing costs.

Business Use Cases:
Targeted Marketing Campaigns
Explanation:
By predicting which customers are most likely to subscribe, the bank can focus its marketing efforts on high-potential leads. This increases the efficiency of campaigns and reduces wasted resources on uninterested customers.
Cost Reduction in Telemarketing
Explanation:
Telemarketing involves significant human resource and time costs. A predictive model helps minimize unnecessary calls by avoiding low-probability clients, leading to reduced operational expenses.
Improved Customer Experience
Explanation:
By avoiding repeated or irrelevant calls to unlikely customers, the bank can enhance the overall customer experience. Clients feel more valued when interactions are relevant and well-timed.
Campaign Performance Optimization
Explanation:
Historical insights from model predictions allow marketing teams to understand what factors influence successful subscriptions. This helps in designing more effective future campaigns with better strategies and timing.
Personalized Financial Product Recommendations
Explanation:
The model can be used as part of a larger recommendation engine that personalizes financial product offerings for clients based on their profile, increasing cross-selling and upselling opportunities. 
Approach:
Step 1: Problem Understanding
Define the business objective: Predict whether a client will subscribe to a term deposit.
Understand the impact of successful predictions on marketing strategy and cost.
Step 2: Data Collection
Use the dataset: bank-additional-full.csv
Explore the metadata and description of each feature.
Step 3: Data Preprocessing
Load data using Pandas.
Check data types, unique values, and distributions.
Encode categorical variables (Label Encoding or One-Hot Encoding).
Handle "unknown" values logically or as separate categories.
Scale/normalize numeric features if needed (e.g., duration, balance).
Step 4: Exploratory Data Analysis (EDA)
Visualize target variable distribution (class balance).
Use bar charts, pie charts, boxplots, and heatmaps.
Analyze relationships between features and the target variable.
Identify any patterns from demographic or campaign features.
Step 5: Feature Engineering
Combine or transform features if useful (e.g., interaction terms).
Create binary flags for known vs. unknown categories.
Drop redundant or irrelevant features if needed.
Step 6: Model Building
Split data into training and testing sets (e.g., 80/20).
Train baseline models: Logistic Regression, Decision Tree.
Try advanced models: Random Forest, XGBoost, LightGBM.
Use cross-validation to avoid overfitting.
Step 7: Model Evaluation
Use evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
Analyze confusion matrix to understand errors.
Compare models and select the best-performing one.
Step 8: Model Saving
Save the trained model using joblib or pickle.
Step 9: Build Streamlit Application
Create a Streamlit app with:
Home Page – Problem intro
EDA Page – Charts and insights
Prediction Page – Input form to predict subscription
Model Info Page – Model performance summary
Step 10: Test and Deploy
Test locally: streamlit run app.py
Deploy via: AWS (EC2)
Step 11: Documentation \& Presentation
Write README.md with project summary and instructions.
Include visual outputs, sample inputs, and predictions.
Optionally, record a video walkthrough.
Results: 
Evaluation Metrics

1. Accuracy
Proportion of total correct predictions.
Formula: (TP + TN) / (TP + TN + FP + FN)
Good for balanced datasets but can be misleading if classes are imbalanced.
2. Precision
How many predicted "yes" are actually "yes".
Formula: TP / (TP + FP)
Important when false positives are costly (e.g., targeting wrong clients).
3. Recall (Sensitivity / True Positive Rate)
How many actual "yes" are correctly predicted.
Formula: TP / (TP + FN)
Important when missing a positive class is costly (e.g., missing potential buyers).
4. F1 Score
Harmonic mean of Precision and Recall.
Formula: 2 × (Precision × Recall) / (Precision + Recall)
Balanced metric when you care about both precision and recall.
5. Confusion Matrix
Gives a matrix of TP, TN, FP, FN.
Helps visually understand the types of errors.
6. ROC Curve (Receiver Operating Characteristic)
Plots True Positive Rate vs. False Positive Rate at different thresholds.
Helps visualize model performance across different thresholds.
7. AUC (Area Under the ROC Curve)
Measures the overall ability of the model to discriminate between classes.
Higher AUC = better model performance.
Project Evaluation metrics:
Define the metrics or criteria used to evaluate the success and effectiveness of the project.
Technical Tags:
Data Understanding \& Domain Insight, Data Preprocessing, Exploratory Data Analysis (EDA), Feature Engineering, Machine Learning Modeling, Model Evaluation, Model Deployment, Streamlit App Development, Version Control \& Documentation, Communication \& Presentation
Data Set:
Data Set: 
Data Set Link: [Data](https://drive.google.com/file/d/1mZNXQHL8xqjPxFZFwcreVZqypzbvf77j/view?usp=sharing) 
Data Set Explanation:
Input variables:
   \# bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services") 
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric) 
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   \# related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   \# other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")
Project Deliverables:
Cleaned \& Preprocessed Dataset
Final dataset used for modeling (with encoding, scaling, etc.)

Exploratory Data Analysis (EDA)
Jupyter notebook with visualizations and insights
Correlation heatmap, distributions, and feature analysis
Model Building Notebook
Code for training and evaluating multiple models
Hyperparameter tuning and performance comparison
Trained Model File
Serialized model (.pkl or .joblib) ready for deployment
Streamlit Web Application
Functional app with pages for Prediction, EDA, and Model Info
Clean UI for user inputs and result display
App Deployment Link (recommended)
Application Deployed to AWS and give the IP Address
Project Report / Summary
Brief document outlining objectives, approach, results, and conclusions
GitHub Repository
Organized codebase with:
README.md (with problem, setup, usage)
requirements.txt
All project files (dataset, notebooks, model, app script)

create a detailed prd in the mdt format for cursor ai

Here is a detailed **PRD in Markdown Table (MDT) format** for Cursor AI, following best practices for Streamlit machine learning apps within banking and predictive classification contexts.[^9][^10][^11]

***

## Product Requirements Document (PRD) - Predictive Term Deposit Subscription Web App

| Section | Description |
| :-- | :-- |
| **Project Overview** | Develop an interactive Streamlit web application for a Portuguese bank using a machine learning model to predict whether a client will subscribe ("yes" or "no") to a term deposit product, supporting more efficient, data-driven marketing campaigns. |
| **Problem Statement** | The bank’s direct marketing campaigns via phone calls result in high resource expenditure with modest subscription rates. The goal is to use historical campaign data and predictive modeling to better identify and target clients who are likely to subscribe, maximizing campaign outcomes and optimizing costs. |
| **Business Use Cases** | - Targeted Marketing: Identify high-potential clients for focused outreach.<br>- Cost Reduction: Limit calls to likely subscribers, minimizing wasted effort.<br>- Enhanced Customer Experience: Reduce irrelevant or repeated calls.<br>- Campaign Optimization: Use model insights to improve future strategies.<br>- Personalized Recommendations: Enable more relevant cross-selling and upselling. |
| **Approach** | 1. Problem Understanding: Define objectives \& impact.<br>2. Data Collection: Use `bank-additional-full.csv`.<br>3. Preprocessing: Data types, encoding, handling unknowns, scaling.<br>4. EDA: Visualize distributions, explore feature relationships.<br>5. Feature Engineering: Interaction terms, binary flags, drop irrelevant data.<br>6. Modeling: Baseline + advanced models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM), with cross-validation.<br>7. Evaluation: Metrics – accuracy, precision, recall, F1, ROC-AUC, confusion matrix.<br>8. Model Saving: Serialize with joblib or pickle.<br>9. Build Streamlit UI: Home, EDA, Prediction, Model Info pages.<br>10. Test and Deploy: Local testing, AWS EC2 deployment.<br>11. Documentation: README, visual outputs, reporting, GitHub repo. |
| **Technical Tags** | Data Understanding, Preprocessing, EDA, Feature Engineering, ML Modeling, Model Evaluation, Deployment, Streamlit UI, Documentation, GitHub, AWS. |
| **Dataset Description** | Input: 16 features—age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome.<br>Target: y ("yes"/"no" subscription to term deposit).<br> Dataset - E:\Python\GuviFinalProject\bank-full.csv
| **Functional Requirements** | - CSV upload and parsing via pandas.<br>- Preprocessing: type check, encoding, scaling, unknowns.<br>- EDA: target/class balance, key feature distributions, demographic/campaign analysis.<br>- Train models: at least 3 algorithms.<br>- Evaluate with confusion matrix, accuracy, recall, precision, F1, ROC, and AUC.<br>- Save trained model.<br>- Streamlit app:<br>  - Home – intro<br>  - EDA – charts/plots<br>  - Prediction – input form, prediction result<br>  - Model Info – metrics and model description<br>- AWS EC2 deployment.<br>- README and project report.<br>- GitHub repository with version control and all files. |
| **Technical Requirements** | - Python 3.8+<br>- Libraries: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, streamlit.<br>- All codebase must follow `.cursorrules` standards.<br>- Data/model security (no external cloud storage except EC2).<br>- Responsive UI for user input and output.<br>- Must support up to 10,000 concurrent users. |
| **Evaluation Metrics** | - Accuracy: \$ (TP + TN) / (TP + TN + FP + FN) \$<br>- Precision: \$ TP / (TP + FP) \$<br>- Recall: \$ TP / (TP + FN) \$<br>- F1 Score: \$ 2 \times \frac{Precision \times Recall}{Precision + Recall} \$<br>- Confusion Matrix: TP, TN, FP, FN counts<br>- ROC Curve \& AUC: TPR vs FPR, area under curve.<br>- User acceptance: functional UI, accurate predictions, app stability. |
| **Acceptance Criteria** | - Data loads and preprocessing runs without error.<br>- Visuals present for all key EDA and performance metrics.<br>- Best model selected using F1/AUC uplift.<br>- All app pages functional in Streamlit.<br>- Accessible via AWS EC2.<br>- GitHub repo includes README, requirements.txt, data, notebooks, deployed app code, trained model.<br>- Clear instructions for setup, testing, and deployment. |
| **Business Logic** | - Treat "unknown" as a valid category during encoding.<br>- Threshold for high-probability prediction is customizable in UI.<br>- Only demographic and campaign-relevant features used.<br>- Flag priority leads in the UI for marketing use. |
| **Deliverables** | - Preprocessed dataset(s)<br>- EDA and modeling Jupyter notebooks<br>- Serialized trained model (.pkl or .joblib)<br>- Complete Streamlit app (app.py)<br>- AWS EC2 deployment link/IP<br>- README.md and documentation<br>- GitHub repo: all files and version history |
| **Glossary** | - Term Deposit: Fixed-term bank saving product.<br>- EDA: Exploratory Data Analysis.<br>- Streamlit: Python web app framework.<br>- Target: Output variable ("y") indicates deposit subscription. |


***

This table can be directly used as a file input for Cursor AI and edited in a markdown-friendly editor. It ensures clarity, modularity, and ease-of-use for AI-powered task automation and development.[^10][^11][^9]
<span style="display:none">[^1][^2][^3][^4][^5][^6][^7][^8]</span>

<div style="text-align: center">⁂</div>

[^1]: https://streamlit.io

[^2]: https://www.kaggle.com/code/aditya2653/basic-banking-system-on-streamlit

[^3]: https://streamlit.io/gallery

[^4]: https://www.ijraset.com/research-paper/streamlit-based-web-application-for-automated-debit-calculation

[^5]: https://www.linkedin.com/posts/himanshu-kumar-b289b226b_simple-bank-app-activity-7355280184750583810-Oz54

[^6]: https://www.youtube.com/watch?v=j54AZjqmCjI

[^7]: https://github.com/streamlit/streamlit

[^8]: https://discuss.streamlit.io/t/a-fun-finance-automation-project-streamlit-app-with-open-cv-login-feature/111682

[^9]: https://github.com/obre10off/cursor_prd_example

[^10]: https://www.chatprd.ai/resources/PRD-for-Cursor

[^11]: https://forum.cursor.com/t/guide-how-to-handle-big-projects-with-cursor/70997

