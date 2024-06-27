# Predicting Cancer Mortality Rates for US Counties
## Description
This project aims to predict cancer mortality rates for US counties using a multivariate Ordinary Least Squares (OLS) regression model. The data for this project has been aggregated from various sources, including the American Community Survey (census.gov), clinicaltrials.gov, and cancer.gov.

## Summary
The project involves building a Multiple Linear Regression model to predict the "TARGET_deathRate" (cancer mortality rate) for US counties. 

## Table of Contents
- Description
- Summary
- Table of Contents
- Installation
- Usage
- Data Description
- Development
- Acknowledgments
- License

## Installation
Clone the repository and install the required dependencies using 
pip install -r requirements.txt

## Usage
To run the Streamlit app:
streamlit run app.py

## Data Description
The data for this project was aggregated from various sources, including the American Community Survey, clinicaltrials.gov, and cancer.gov. The dataset contains various features that may influence cancer mortality rates, such as demographic information, socioeconomic factors, and clinical trial data.

## Development
## Data Cleaning
The data was cleaned and preprocessed to remove any inconsistencies and missing values. This step involved handling missing data, transforming variables, and ensuring the dataset was suitable for regression analysis.

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) was conducted to understand the distribution of variables, identify any potential relationships, and detect outliers. Key insights from the EDA included the identification of important features that might influence cancer mortality rates.

## Model Building and Evaluation
A multivariate OLS regression model was built to predict the TARGET_deathRate. The model was trained on a subset of the data, and its performance was evaluated using metrics such as adjusted R-squared and Root Mean Squared Error (RMSE).

## Acknowledgments
Credit to the American Community Survey (census.gov), clinicaltrials.gov, and cancer.gov for providing the data used in this project.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.




