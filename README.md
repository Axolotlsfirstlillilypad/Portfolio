# Repository
Portfolio of Data Science Projects showcasing my work in machine learning, data visualization, NLP, and data engineering. Includes detailed models, code explanations, and interactive dashboards. Technologies used: R, TensorFlow, ggplot, XGBoost. Aimed at potential employers and collaborators.

Employer Highlight: Full Portfolio

This portfolio showcases ten diverse data science projects that demonstrate a wide range of analytical, modeling, and visualization skills highly relevant to employers. Across structured and unstructured datasets, the projects illustrate the ability to clean and preprocess complex data, perform exploratory data analysis (EDA), build predictive and classification models, evaluate model performance with appropriate metrics, and create insightful visualizations for decision-making.

The portfolio emphasizes both practical business applications and technical proficiency:

Airbnb Price Prediction – Predict rental prices using structured real estate data, demonstrating regression modeling, feature engineering, and visualization to inform pricing strategies.

COVID-19 Analysis & Dashboard – Analyze and visualize pandemic trends globally, create interactive dashboards, and handle time series data to support public health decision-making.

Sales Forecasting – Generate monthly sales forecasts using ARIMA models, showing time series analysis skills and predictive accuracy evaluation.

Customer Segmentation – Apply K-means clustering to retail data, segmenting customers based on purchasing behavior to guide targeted marketing strategies.

Customer Churn Modeling – Predict churn using Random Forests, evaluate performance with F1-score, and identify key factors driving retention, supporting CRM initiatives.

Sentiment Analysis – Process text data and classify customer reviews with XGBoost, demonstrating NLP preprocessing, vectorization, and classification on unstructured data.

E-Commerce Web Scraping – Collect, clean, and analyze product data from online sources, visualize pricing, availability, and ratings, highlighting automated data collection and market analysis capabilities.

Iris Prediction API – Develop and deploy a RESTful API for real-time predictions, showing practical skills in model serving and software integration.

HR Attrition Prediction – Predict employee turnover using XGBoost, handling class imbalance and interpreting feature importance to inform HR decision-making.

Credit Card Fraud Detection – Detect fraudulent transactions using XGBoost, apply imbalanced classification techniques, and measure performance with F1-score and AUC, critical for risk management.

# Portfolio of 10 Data Science & Machine Learning Projects

This repository contains **10 diverse projects** covering data analysis, machine learning, forecasting, clustering, natural language processing, web scraping, API creation, and more. Each project includes **R code, data preprocessing, modeling, and visualizations**, showcasing practical applications.

# Portfolio Folder Structure
```markdown
Portfolio-Repository/
├── README.md
├── data/
│   ├── Listings.csv
│   ├── owid-covid-data.csv
│   ├── sales.csv
│   ├── mall_customers.csv
│   ├── churn.csv
│   ├── reviews.csv
│   ├── books_scraped.csv
│   ├── HR-Employee-Attrition.csv
│   └── creditcard.csv
├── imgs/
│   ├── project1_price_plot.png
│   ├── project2_covid_plot.png
│   ├── project3_forecast_plot.png
│   ├── project4_cluster_plot.png
│   ├── project5_churn_feature_importance.png
│   ├── project6_sentiment_plot.png
│   ├── project7_price_rating_plot.png
│   ├── project8_api_test_results.png
│   ├── project9_hr_feature_importance.png
│   └── project10_fraud_feature_importance.png
├── Project1_AirbnbPrice/
│   └── scripts/
│       └── airbnb_price_prediction.R
├── Project2_COVIDShiny/
│   └── scripts/
│       └── covid_shiny_app.R
├── Project3_SalesForecast/
│   └── scripts/
│       └── sales_forecast.R
├── Project4_MallClustering/
│   └── scripts/
│       └── mall_clustering.R
├── Project5_ChurnPrediction/
│   └── scripts/
│       └── churn_model.R
├── Project6_SentimentXGBoost/
│   └── scripts/
│       └── sentiment_xgboost.R
├── Project7_BookScraping/
│   └── scripts/
│       └── book_scrape_analysis.R
├── Project8_IrisAPI/
│   └── scripts/
│       └── plumber_api.R
├── Project9_HRAttritionXGBoost/
│   └── scripts/
│       └── hr_attrition_xgboost.R
└── Project10_CreditCardFraud/
    └── scripts/
        └── creditcard_fraud_xgboost.R

```
---

## 1. Airbnb Price Prediction

**Goal:** Predict Airbnb listing prices using property features.

**Description:**  
- Loaded Airbnb listings CSV.  
- Cleaned dataset: removed duplicates and rows with missing `price` or `bedrooms`.  
- Split into training/testing sets.  
- Built two models:  
  - **Linear Regression** (`caret`)  
  - **Random Forest** (`randomForest`)  
- Compared performance using **RMSE**.  
- Visualized actual vs predicted prices.  
- Plotted variable importance from Random Forest.

**Key Insight:** Random Forest generally captures non-linear relationships better, often outperforming linear regression in RMSE.

---

## 2. COVID-19 Interactive Dashboard

**Goal:** Explore COVID-19 trends and visualize daily cases.

**Description:**  
- Loaded global COVID-19 dataset.  
- Cleaned and filtered country-level data.  
- Performed EDA: top 10 countries by total cases, daily new cases for USA and top countries.  
- Plotted interactive visualizations using **Plotly**.  
- Built **Shiny app** to select a country and see daily new cases dynamically.

**Key Insight:** Visualizations highlight how trends vary by country and over time, helping identify peaks and surges.

---

## 3. Sales Forecasting with ARIMA

**Goal:** Forecast future monthly sales using time series modeling.

**Description:**  
- Generated synthetic monthly sales data (2018–2022).  
- Converted data to a time series object (`ts`).  
- Fitted **ARIMA model** (`forecast::auto.arima`).  
- Forecasted next 12 months of sales.  
- Plotted forecast with confidence intervals.

**Key Insight:** ARIMA captures seasonal patterns and trends for short-term forecasting.

---

## 4. Customer Segmentation (K-Means Clustering)

**Goal:** Segment mall customers for targeted marketing.

**Description:**  
- Loaded customer dataset with `Age`, `AnnualIncome`, `SpendingScore`.  
- Standardized features.  
- Applied **K-means clustering** (5 clusters).  
- Added cluster labels to original data.  
- Visualized clusters with `factoextra::fviz_cluster`.

**Key Insight:** Clusters reveal distinct customer groups based on spending behavior and demographics.

---

## 5. Customer Churn Prediction

**Goal:** Predict which customers are likely to churn.

**Description:**  
- Generated synthetic churn dataset.  
- Split data into train/test.  
- Trained **Random Forest model** to predict churn.  
- Evaluated using **confusion matrix** and **F1-score**.  
- Plotted variable importance.  
- Visualized churn distribution.

**Key Insight:** Tenure, monthly charges, and total charges are key predictors of churn.

---

## 6. Sentiment Analysis on Product Reviews (XGBoost)

**Goal:** Classify reviews as positive or negative.

**Description:**  
- Loaded review dataset.  
- Preprocessed text using `text2vec` (tokenization + TF-IDF).  
- Split data into train/test.  
- Converted labels to numeric for XGBoost.  
- Trained **XGBoost classifier**.  
- Predicted and evaluated using **F1-score** for positive sentiment.

**Key Insight:** TF-IDF + XGBoost is effective for sentiment classification, handling sparse text data well.

---

## 7. E-Commerce Analysis: Books to Scrape

**Goal:** Scrape and analyze online book data.

**Description:**  
- Scraped book titles, prices, availability, ratings from first page.  
- Analyzed:  
  - Price distribution  
  - Stock availability  
  - Price vs rating  
  - Categorized books as Budget / Mid-range / Premium  
- Saved cleaned dataset for future analysis.

**Key Insight:** Data scraping combined with visualization allows insights on pricing, stock, and quality metrics.

---

## 8. Iris Flower Prediction API (Plumber)

**Goal:** Expose ML model as an API.

**Description:**  
- Trained a **decision tree** on Iris dataset.  
- Created **Plumber API** to predict flower species based on sepal/petal measurements.  
- API endpoint: `/predict` accepts 4 features and returns species prediction.

**Key Insight:** Demonstrates deployment of an ML model as a RESTful API.

---

## 9. Employee Attrition Prediction (XGBoost)

**Goal:** Predict employee attrition.

**Description:**  
- Loaded HR employee dataset.  
- Preprocessed: factor encoding, dropped identifiers, handled class imbalance.  
- Split into train/test.  
- Trained **XGBoost classifier**.  
- Evaluated with **F1-score** and **AUC**.  
- Plotted feature importance.

**Key Insight:** Features like job satisfaction and tenure are strong predictors of attrition.

---

## 10. Credit Card Fraud Detection (XGBoost)

**Goal:** Detect fraudulent credit card transactions.

**Description:**  
- Loaded credit card dataset.  
- Preprocessed: factor conversion, one-hot encoding, handled class imbalance.  
- Split into train/test.  
- Trained **XGBoost classifier**.  
- Predicted fraud, evaluated using **F1-score** and **AUC**.  
- Plotted feature importance.

**Key Insight:** Handling class imbalance with weighted XGBoost improves detection of rare fraud cases.

---

## Technologies & Libraries Used

- **Data Manipulation:** `tidyverse`, `dplyr`, `janitor`  
- **Machine Learning:** `caret`, `randomForest`, `xgboost`  
- **Time Series & Forecasting:** `forecast`, `lubridate`  
- **Clustering & EDA:** `cluster`, `factoextra`, `ggplot2`  
- **Text Analysis:** `text2vec`, `Matrix`  
- **Web Scraping:** `rvest`, `stringr`  
- **Visualization:** `ggplot2`, `plotly`  
- **Shiny Web Apps & API:** `shiny`, `plumber`  



# Project 1: Airbnb Price Prediction

## Employer Highlight
This project demonstrates the ability to predict rental prices using structured real estate data. Predicting rental prices is valuable for companies in property management, travel, or online marketplaces. The project showcases skills in data cleaning, exploratory data analysis (EDA), predictive modeling, evaluation, and visualization. These skills are critical for evidence-based decision making in a business context.

## Code Walkthrough & Detailed Discussion

## 1. Airbnb Price Prediction

### Employer Highlight
This project demonstrates the ability to predict rental prices using structured real estate data, which is valuable for companies in property management, travel, or online marketplaces. It showcases skills in data cleaning, exploratory analysis, predictive modeling, evaluation, and visualization — all critical for evidence-based business decisions.

---

### Code Walkthrough & Detailed Discussion

#### Packages
```r
library(tidyverse)
library(janitor)
library(caret)
library(randomForest)
``` 
We load tidyverse for data manipulation, visualization, and wrangling, and janitor for cleaning column names and ensuring consistency. Caret and randomforest make it easier to coax the model into a usable form.
```r
library(readr)
airbnb <- read_csv(unz("C:/Users/User/Downloads/archive (1).zip", "Listings.csv"))
```
We read the Airbnb dataset directly from a ZIP file. This can be done by using unz(), then using the path of the zip folder and the name of the intended file as arguments of unz(). 
```r
glimpse(airbnb)
summary(airbnb)
```
We examine data types and summary statistics.
glimpse() shows which variables are numeric vs categorical, while summary() highlights price spread, outliers, and skewness. Also it suses out the NA variables, which we drop. 
```r
airbnb_model <- drop_na(airbnb)
summary(airbnb_model)
```
```r
set.seed(123)
trainIndex <- createDataPartition(airbnb_model$price, p = 0.8, list = FALSE)
train <- airbnb_model[trainIndex, ]
test <- airbnb_model[-trainIndex, ]
```
We split the data 80/20 into training and testing sets to evaluate model performance on unseen data.
set.seed(123) ensures reproducibility.
Given the right-skewed price distribution, this model may not capture non-linearities — motivating our exploration of Random Forest. 
```r
model_rf <- randomForest(
  price ~ bedrooms + bathrooms + accommodates,
  data = train,
  ntree = 500,
  importance = TRUE
)
```
RMSE (Root Mean Square Error) is the best metric because the target variable (price) is continuous and RMSE penalizes large deviations in prices.

Random Forest typically achieves a lower RMSE, confirming its robustness for this dataset.
```r
pred_lm <- predict(model_lm, test)
pred_rf <- predict(model_rf, test)

lm_rmse <- RMSE(pred_lm, test$price)
rf_rmse <- RMSE(pred_rf, test$price)

cat("Linear Model RMSE:", lm_rmse, "\n")
cat("Random Forest RMSE:", rf_rmse, "\n")
```
We use a scatter plot to depict the discrete points to check if the actual and predicted are close. They are, so great. 
```r
results <- tibble(
  actual = test$price,
  predicted = pred_rf
)

ggplot(results, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted Prices",
       x = "Actual Price", y = "Predicted Price")
```

Next Steps:

Incorporate location, reviews, and amenities as predictors for improved accuracy.

Experiment with XGBoost or Gradient Boosting.

Develop a Shiny dashboard for interactive price predictions.
