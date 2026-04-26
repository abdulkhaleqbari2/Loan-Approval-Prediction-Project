🏦 Loan Approval Prediction — Machine Learning Classification

A supervised machine learning project that predicts whether a loan application will be approved based on applicant financial and demographic data, covering the full ML pipeline from EDA and feature engineering to model evaluation.


📌 Project Overview
This project applies supervised classification to a real-world Loan Approval Dataset to help financial institutions automate and improve their loan eligibility decisions. The notebook walks through end-to-end data science — exploratory analysis, data cleaning, feature engineering, encoding, scaling, and comparison of classification models.

🎯 Objective
Predict loan approval status (Loan_Status: Approved or Rejected) based on applicant profile data, and identify the most accurate classification model for deployment.

📂 Dataset

File: Loan_Data.csv
Domain: Banking / Financial Services
Task: Binary Classification

FeatureDescriptionLoan_IDUnique loan identifier (dropped before modeling)GenderMale / FemaleMarriedMarital statusDependentsNumber of dependentsEducationGraduate / Not GraduateSelf_EmployedSelf-employment statusApplicantIncomeMonthly income of applicantCoapplicantIncomeMonthly income of co-applicantLoanAmountRequested loan amountLoan_Amount_TermRepayment term in monthsCredit_HistoryCredit history track record (1 = Good, 0 = Bad)Property_AreaUrban / Semiurban / RuralLoan_Status✅ Target variable (Approved / Rejected)


Language: Python 3
Libraries: numpy, pandas, matplotlib, scikit-learn


🔍 Project Workflow
1. Exploratory Data Analysis (EDA)

Credit History vs Loan Status cross-tabulation — confirmed that applicants with good credit history (Credit_History = 1) have significantly higher approval rates
Income Distribution: ApplicantIncome and CoapplicantIncome both found to be right-skewed with heavy outliers
Boxplots by Education: Graduate applicants show noticeably higher incomes than non-graduates
LoanAmount Analysis: Right-skewed with outliers; identified need for transformation

2. Data Preprocessing & Cleaning

Missing Value Treatment: Handled nulls across multiple columns using appropriate strategies:

Categorical columns (Gender, Married, Dependents, Self_Employed, Loan_Amount_Term, Credit_History) → filled with mode
Numerical column (LoanAmount) → filled with mean


Log Transformation: Applied np.log() to LoanAmount to reduce skewness and normalize the distribution — confirmed visually with histograms

3. Feature Engineering

TotalIncome: Combined ApplicantIncome and CoapplicantIncome into a single feature to reduce dimensionality and capture household earning power
Log Transform on TotalIncome: Applied np.log() to further normalize the income distribution
Dropped: ApplicantIncome, CoapplicantIncome, and Loan_ID (identifier — no predictive value)

4. Encoding & Scaling

Label Encoding: Encoded categorical columns (Gender, Married, Dependents, Education, Self_Employed, Property_Area) using LabelEncoder
Manual Mapping: Gender → {Male: 1, Female: 0}; Property_Area → {Urban: 1, Rural: 0, Semiurban: 2}
Feature Scaling: Applied StandardScaler to normalize all features before model training

5. Model Training & Evaluation
Two classification models trained and compared:
ModelNotesDecision Tree ClassifierTrained with criterion='entropy'; interpretable, rule-based predictionsGaussian Naïve BayesProbabilistic classifier; fast, effective on smaller datasets
Each model evaluated using Accuracy Score from sklearn.metrics.

📊 Key Insights

✅ Credit History is the single most influential feature — applicants with credit history are far more likely to get approved
📈 Income skewness must be addressed before modeling; log transformation significantly improves model performance
🎓 Education level has a visible impact on income, indirectly affecting loan amounts and approval rates
🏠 Property Area and Married status serve as meaningful demographic signals for loan risk




📁 Project Structure
loan-approval-prediction/
│
├── Project7.ipynb           # Main Jupyter Notebook
├── README.md                # Project documentation
└── data/
    └── Loan_Data.csv        # Dataset (add locally)

💡 Key Learnings

Credit history is the dominant predictor in loan approval — reinforces real-world banking intuition
Log transformation is a powerful technique for handling income and financial data skewness
Combining applicant and co-applicant income into a single TotalIncome feature reduces noise and improves model clarity
Naïve Bayes is a strong baseline for binary classification on mixed-feature tabular datasets
