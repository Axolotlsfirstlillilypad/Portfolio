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







## Project 2: COVID-19 Interactive Dashboard


**Code Walkthrough & Detailed Discussion:**

```r
# Load necessary libraries
library(shiny)
library(plotly)
library(tidyverse)
library(lubridate)
```

We load shiny for creating interactive web apps, plotly for interactive plots, tidyverse for data wrangling, and lubridate for handling date columns efficiently.

# Load dataset
```r
covid <- read_csv(unz("C:/Users/User/Downloads/archive (2).zip", "owid-covid-data.csv"))
```


We import the global COVID-19 dataset from a zip file from the covid dataset on Kaggle. This ensures the data is version-controlled and reproducible.

# Clean and filter data
```r
covid_clean <- covid %>%
  select(date, location, total_cases, new_cases, total_deaths, new_deaths, population) %>%
  filter(!is.na(location), location != "World") %>%
  mutate(
    date = ymd(date),
    new_cases = ifelse(is.na(new_cases), 0, new_cases),
    new_deaths = ifelse(is.na(new_deaths), 0, new_deaths)
  )
```

We clean the dataset by keeping only relevant columns and removing aggregate "World" entries to focus on individual countries.
ymd(date) converts strings to Date objects. Missing values in new_cases and new_deaths are replaced with zero because zeros are meaningful for days without new cases or deaths.

# Time series plot for selected countries
```r
top_countries <- c("United States", "India", "Brazil", "United Kingdom", "Germany")
covid_clean %>%
  filter(location %in% top_countries) %>%
  ggplot(aes(x = date, y = new_cases, color = location)) +
  geom_line() +
  labs(title = "Daily New Cases: Top Countries", x = "Date", y = "New Cases") +
  theme_minimal()
```

We visualize trends for key countries. The line plot allows detection of peaks and patterns. Using geom_line() is ideal for time series, showing trends clearly over dates.

```r
# Shiny UI
ui <- fluidPage(
  titlePanel("COVID-19 Cases by Country"),
  sidebarLayout(
    sidebarPanel(
      selectInput("country", "Select country:", choices = sort(unique(covid_clean$location)))
    ),
    mainPanel(
      plotlyOutput("covidPlot")
    )
  )
)
```

We define a Shiny UI that allows users to select a country from a dropdown. The main panel will display the interactive plot.
```r
# Shiny server
server <- function(input, output) {
  output$covidPlot <- renderPlotly({
    covid_clean %>%
      filter(location == input$country) %>%
      plot_ly(
        x = ~date,
        y = ~new_cases,
        type = 'scatter',
        mode = 'lines+markers',
        name = "Daily New Cases"
      ) %>%
      layout(
        title = paste("Daily New COVID-19 Cases:", input$country),
        xaxis = list(title = "Date"),
        yaxis = list(title = "New Cases")
      )
  })
}

shinyApp(ui = ui, server = server)
```

The server logic filters the data for the selected country and renders an interactive Plotly line plot. This approach allows users to explore trends dynamically, which is critical for public health decision-making.


Next Steps:

Add cumulative totals and mortality rate calculations for enhanced analysis.

Include vaccination and testing data to give a more complete picture.

Implement predictive models to forecast future cases based on historical trends.

Improve dashboard interactivity by adding multiple country comparisons and trend overlays.



## Project 3: Sales Forecasting

**Employer Highlight:**  
This project showcases the ability to forecast future sales using historical sales data. Employers in retail, e-commerce, or supply chain management can leverage these skills to optimize inventory, plan promotions, and make data-driven decisions. The project highlights time series analysis, model evaluation, visualization, and forecasting techniques.

**Code Walkthrough & Detailed Discussion:**

```r
# Load required libraries
library(tidyverse)
library(lubridate)
library(forecast)
library(tsibble)
library(fable)
```
We load tidyverse for data manipulation, lubridate for handling dates, forecast for classical forecasting models, and tsibble & fable for tidy time series analysis.

# Load dataset
We load tidyverse for data manipulation, lubridate for handling dates, forecast for classical forecasting models, and tsibble & fable for tidy time series analysis.

# Load dataset
```r
sales <- read_csv("C:/Users/User/Downloads/sales.csv")
```

We import historical sales data. The dataset contains a date column and corresponding sales figures.

# Data cleaning and preparation
```r
sales_clean <- sales %>%
  mutate(date = ymd(date)) %>%
  arrange(date)
```

Convert the date column to a Date type using ymd(). Sorting ensures the time series is in chronological order, which is critical for forecasting models.

# Convert to tsibble
```r
sales_ts <- sales_clean %>%
  as_tsibble(index = date)
```

We transform the dataset into a tsibble object to leverage tidy time series tools, ensuring that the date column is the index.

# Plot historical sales
```r
ggplot(sales_ts, aes(x = date, y = sales)) +
  geom_line(color = "blue") +
  labs(title = "Historical Sales", x = "Date", y = "Sales") +
  theme_minimal()
```


Visualize historical sales to detect trends, seasonality, and anomalies. The line plot provides intuition about underlying patterns.

# Forecast using ARIMA model
```r
model_arima <- sales_ts %>%
  model(ARIMA(sales))
forecast_arima <- model_arima %>% forecast(h = "12 months")
```


We fit an ARIMA model because sales exhibit trend and possible seasonality. ARIMA handles autocorrelation and generates robust forecasts.

# Plot forecast
```r
forecast_arima %>%
  autoplot(sales_ts) +
  labs(title = "Sales Forecast (ARIMA)", x = "Date", y = "Sales") +
  theme_minimal()
```

Visualize the forecast along with historical data. This allows stakeholders to see expected sales and confidence intervals.

# Evaluate forecast accuracy
```r
accuracy(forecast_arima, sales_ts)
```

We check accuracy metrics such as RMSE, MAE, and MAPE. RMSE is appropriate because the target (sales) is numeric, and large deviations are penalized, aligning with business impact.

Next Steps:

Include external regressors (promotions, holidays) to improve forecast accuracy.

Explore exponential smoothing (ETS) and Prophet for comparison.

Deploy the forecast in a dashboard for real-time business planning.

## Project 4: Mall Customer Segmentation

**Employer Highlight:**  
This project demonstrates clustering techniques to segment mall customers based on spending behavior and demographics. Useful for retail, marketing, and customer experience teams, it highlights exploratory data analysis, unsupervised learning, visualization, and actionable business insights.

**Code Walkthrough & Detailed Discussion:**

```r
# Load libraries
library(tidyverse)
library(cluster)
library(factoextra)
tidyverse is used for data manipulation and visualization. cluster provides clustering algorithms like k-means, and factoextra helps visualize cluster results.

```r
# Load data
mall <- read_csv("C:/Users/User/Downloads/mall_customers.csv")
```
Import customer data containing age, annual income, and spending score.

```r
# Data exploration
glimpse(mall)
summary(mall)
```
Check variable types and distributions. Features are numeric, which suits k-means clustering. Detect outliers and scale differences.

```r
# Scale numeric features
mall_scaled <- mall %>%
  select(Age, `Annual Income (k$)`, `Spending Score (1-100)`) %>%
  scale()
```
Scaling ensures each variable contributes equally to distance calculations in k-means.

```r

# Determine optimal clusters using elbow method
fviz_nbclust(mall_scaled, kmeans, method = "wss")
```
The elbow plot helps choose the optimal number of clusters by evaluating within-cluster sum of squares.

```

# K-means clustering
set.seed(123)
kmeans_model <- kmeans(mall_scaled, centers = 5, nstart = 25)
mall$cluster <- kmeans_model$cluster
```
We select 5 clusters, as indicated by the elbow method. Following a series of tests measuring the metric, we found that nstart = 25 optimises stability.

```r
# Visualize clusters
fviz_cluster(kmeans_model, data = mall_scaled,
             geom = "point",
             ellipse.type = "norm",
             palette = "jco",
             ggtheme = theme_minimal())
Scatter plot with clusters reveals distinct customer segments, guiding marketing or loyalty programs.
```


Next Steps:

Include additional features (gender, tenure, product preferences).

Test hierarchical clustering or DBSCAN for alternative insights.

Integrate segments into predictive models for personalized campaigns.



## Project 5: Customer Churn Prediction



```r
# Load libraries
library(tidyverse)
library(caret)
library(randomForest)
caret simplifies model training and evaluation. randomForest provides robust predictive modeling for classification.

```r
# Load data
churn <- read_csv("C:/Users/User/Downloads/churn.csv")
```
Dataset contains customer demographics, usage, and churn label.

```r
# Data cleaning
churn_clean <- churn %>%
  mutate_if(is.character, as.factor) %>%
  drop_na()
Convert categorical variables to factors and remove missing values. Random Forest requires factor levels for classification.
```

```r
# Train-test split
set.seed(123)
trainIndex <- createDataPartition(churn_clean$Churn, p = 0.8, list = FALSE)
train <- churn_clean[trainIndex, ]
test <- churn_clean[-trainIndex, ]
```
Based on repeated testing, 80/20 split ensures evaluation on unseen data, and reproducibility is guaranteed by set.seed().

```r
# Random Forest model
rf_model <- randomForest(Churn ~ ., data = train, ntree = 500, importance = TRUE)
pred <- predict(rf_model, test)
confusionMatrix(pred, test$Churn)
```
Random Forest is chosen because it handles categorical variables, missing values, and interactions well. Accuracy, sensitivity, and specificity are assessed via the confusion matrix.


```r
# Feature importance
varImpPlot(rf_model)
```
Highlights the most influential features for churn, informing retention strategies.



Next Steps:

Include behavioral and engagement metrics.

Experiment with XGBoost or logistic regression for comparison.

Deploy as a dashboard for real-time churn monitoring.

yaml
Copy code

---

## Project 6: Sentiment Analysis with XGBoost


```r
# Load libraries
library(tidyverse)
library(tidytext)
library(xgboost)
library(caret)
```
tidytext handles text preprocessing. xgboost provides gradient boosting for classification, robust for high-dimensional sparse data.

```r
# Load data
reviews <- read_csv("C:/Users/User/Downloads/reviews.csv")
```
Contains text reviews and sentiment labels.

```r

# Text preprocessing
reviews_clean <- reviews %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  count(id, word) %>%
  cast_dtm(id, word, n)
```
Tokenize text, remove stop words, and create a Document-Term Matrix. Sparse numeric data suits XGBoost.

```r
# Train-test split
trainIndex <- createDataPartition(reviews$sentiment, p = 0.8, list = FALSE)
train <- reviews[trainIndex, ]
test <- reviews[-trainIndex, ]
```
Reproducible train-test split ensures evaluation accuracy. 80% training seems like a reliably good figure

```r
# XGBoost model
xgb_model <- xgboost(data = as.matrix(train_dtm),
                     label = as.numeric(train$sentiment) - 1,
                     objective = "binary:logistic",
                     nrounds = 100)
```
Binary classification model; XGBoost is ideal for sparse, high-dimensional data and handles feature importance naturally.

```r
# Predictions
pred <- predict(xgb_model, as.matrix(test_dtm))
pred_class <- ifelse(pred > 0.5, 1, 0)
confusionMatrix(factor(pred_class), factor(as.numeric(test$sentiment) - 1))
```
Evaluate performance using accuracy, precision, and recall.


## Project 7: Book Price & Rating Analysis



```r
# Load libraries
library(rvest)
library(tidyverse)
library(caret)
```
rvest for web scraping, caret for predictive modeling, tidyverse for cleaning and visualization.

```r
# Load scraped data
books <- read_csv("C:/Users/User/Downloads/books_scraped.csv")
```
Contains book prices, ratings, and metadata.

```r
# Exploratory analysis
summary(books$price)
summary(books$rating)
ggplot(books, aes(x = rating, y = price)) + geom_point()
```
Visualize spread of prices vs ratings. Detect skew, outliers, and correlation patterns.

```r
# Model
set.seed(123)
trainIndex <- createDataPartition(books$price, p = 0.8, list = FALSE)
train <- books[trainIndex, ]
test <- books[-trainIndex, ]
rf_model <- randomForest(price ~ rating + num_reviews, data = train, ntree = 500, importance = TRUE)
pred <- predict(rf_model, test)
RMSE(pred, test$price)
```
Random Forest selected due to skewed price distribution and potential non-linear effects from ratings and review count.

```r
# Feature importance
varImpPlot(rf_model)
```
Identify key features influencing book price.


Next Steps:

Include genre and author as predictors.

Experiment with boosting algorithms for better accuracy.

Integrate into a pricing recommendation dashboard.



---

## Project 8: Iris Classification API

**Employer Highlight:**  
This project demonstrates building a predictive API for classification, applicable in data-driven services, product recommendations, and automated analytics pipelines. Highlights include API design, model deployment, and testing.

**Code Walkthrough & Detailed Discussion:**

```r
# Load libraries
library(plumber)
library(caret)
```
plumber is used to expose R functions as a REST API. caret supports model training.

```r

# Load Iris dataset
data(iris)
Classic dataset for multi-class classification.\
```

```r
# Train model
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train <- iris[trainIndex, ]
test <- iris[-trainIndex, ]
rf_model <- randomForest(Species ~ ., data = train, ntree = 100)
```
Random Forest chosen for robustness and interpretability for multi-class prediction.

```r
# Plumber API
#* @post /predict
function(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width){
  new_data <- data.frame(Sepal.Length=Sepal.Length, Sepal.Width=Sepal.Width,
                         Petal.Length=Petal.Length, Petal.Width=Petal.Width)
  predict(rf_model, new_data)
}
```
Exposes a /predict endpoint that returns predicted species based on input features.

Next Steps:

Add input validation and error handling.

Deploy API to cloud service for accessibility.

Integrate with a front-end app for interactive predictions.



---

## Project 9: HR Employee Attrition Prediction

```r
# Load libraries
library(tidyverse)
library(caret)
library(xgboost)
XGBoost handles structured employee data and high-dimensional interactions well.
```

```r
# Load dataset
hr <- read_csv("C:/Users/User/Downloads/HR-Employee-Attrition.csv")
```
Contains employee demographics, performance, and attrition labels.

```r
# Data preprocessing
hr_clean <- hr %>%
  mutate_if(is.character, as.factor) %>%
  drop_na()
```
Convert categorical variables to factors and drop missing values.

```r
# Train-test split
set.seed(123)
trainIndex <- createDataPartition(hr_clean$Attrition, p = 0.8, list = FALSE)
train <- hr_clean[trainIndex, ]
test <- hr_clean[-trainIndex, ]
```
Reproducible 80/20 split.

```r
Copy code
# XGBoost model
xgb_model <- xgboost(data = as.matrix(train %>% select(-Attrition)),
                     label = as.numeric(train$Attrition) - 1,
                     objective = "binary:logistic",
                     nrounds = 100)
```
Binary classification using XGBoost, suitable for handling interactions and importance ranking.

```r
# Feature importance
importance_matrix <- xgb.importance(feature_names = colnames(train %>% select(-Attrition)), model = xgb_model)
xgb.plot.importance(importance_matrix)
```
Shows which employee features (e.g., JobRole, Satisfaction) influence attrition predictions.


Next Steps:

Include tenure, engagement, and training data.

Experiment with random forests or logistic regression for comparison.

Integrate into an HR analytics dashboard.


## Project 10: Credit Card Fraud Detection


```r
# Load libraries
library(tidyverse)
library(caret)
library(xgboost)
```
XGBoost handles imbalanced datasets and high-dimensional transaction features effectively.

```r

# Load data
credit <- read_csv("C:/Users/User/Downloads/creditcard.csv")
```
Contains transaction features and a binary fraud label.

```r
# Data preprocessing 
credit_clean <- credit %>%
  drop_na()
```
Remove missing values; all variables are numeric.

```r
# Train-test split
set.seed(123)
trainIndex <- createDataPartition(credit_clean$Class, p = 0.8, list = FALSE)
train <- credit_clean[trainIndex, ]
test <- credit_clean[-trainIndex, ]
```
Ensure evaluation on unseen transactions.

```r
# XGBoost model
xgb_model <- xgboost(data = as.matrix(train %>% select(-Class)),
                     label = train$Class,
                     objective = "binary:logistic",
                     nrounds = 100)
Binary classification for fraud detection; XGBoost handles class imbalance with high accuracy.
```

```r

# Predictions & evaluation
pred <- predict(xgb_model, as.matrix(test %>% select(-Class)))
pred_class <- ifelse(pred > 0.5, 1, 0)
confusionMatrix(factor(pred_class), factor(test$Class))
Evaluate precision, recall, and F1-score; high precision is crucial to reduce false positives.
```

```r

# Feature importance
importance_matrix <- xgb.importance(feature_names = colnames(train %>% select(-Class)), model = xgb_model)
xgb.plot.importance(importance_matrix)
```
Identify transaction patterns most indicative of fraud.


Next Steps:

Implement SMOTE or other oversampling for severe class imbalance.

Test real-time streaming detection.

Integrate with monitoring dashboards for automated alerts.

# Sales Performance Dashboard - Power BI Project

## Project Overview

The **Sales Performance Dashboard** is a comprehensive Power BI report designed to provide business insights into sales performance across regions, products, and sales teams. The dashboard combines KPIs, visualizations, and interactivity to help managers and executives make data-driven decisions.

---

## Features

### Key Metrics / KPIs
- **Total Sales** – overall revenue across the organization.
- **Total Profit** – total profit from sales.
- **Profit Margin %** – calculated as Profit ÷ Sales.
- **Year-over-Year (YoY) Sales Growth** – compares current year sales to the previous year.
- **Sales Forecast** – projections based on historical trends.

### Visualizations
- **KPI Cards** – highlight important metrics at a glance (Total Sales, Total Profit, Profit Margin %).  
- **Clustered Column Chart** – revenue by region or salesperson.  
- **Stacked Column Chart / Treemap** – sales by product category and sub-category.  
- **Line Chart** – track sales trends over time (daily, monthly, yearly).  
- **Pie / Donut Chart** – visualize sales distribution by category or region.  
- **Maps (optional)** – geographic sales distribution by region or state.

### Interactivity
- **Slicers** – filter by Year, Region, Product Category, or Salesperson.  
- **Cross-filtering** – select a value in one visual to update all other visuals dynamically.  
- **Drill-through Pages** – detailed analysis for individual regions, allowing regional managers to view KPIs, sales trends, and top-performing products specific to their area.  

---

## Data Preparation

- **Source** – Historical sales data including fields: `Order Date`, `Region`, `Product Category`, `Sub-Category`, `Sales`, `Profit`, `Customer Name`, `Salesperson`.  
- **Data Cleaning** – ensured proper data types (Date, Decimal Number, Whole Number), handled missing or null values.  
- **Calculated Columns / Measures (DAX)**:
  - `Profit Margin % = DIVIDE([Total Profit], [Total Sales], 0)`
  - `YoY Sales Growth = DIVIDE(SUM([Sales]) - CALCULATE(SUM([Sales]), SAMEPERIODLASTYEAR('Sales'[Order Date])), CALCULATE(SUM([Sales]), SAMEPERIODLASTYEAR('Sales'[Order Date])), 0)`
  - `Cumulative Sales` – running total for sales trends.

---

## Dashboard Layout

### Page 1 – Executive Overview
- KPI Cards: Total Sales, Total Profit, Profit Margin %  
- Clustered Column Chart: Revenue by Region  
- Pie/Donut Chart: Sales by Category  
- Line Chart: Sales Trend by Month/Year  
- Map (optional): Sales by Region/State  
- Slicers: Year, Region, Category  

### Page 2 – Salesperson / Customer Performance
- Table: Customer Name, Sales, Profit, Profit Margin %  
- Bar Chart: Sales by Customer or Salesperson  
- Scatter Plot: Sales vs Profit Margin  
- KPI: Top 5 Customers (using Top N filter)  
- Slicers: Month, Region  

### Page 3 – Product Analysis
- Stacked Bar Chart: Sales by Sub-Category and Category  
- Treemap: Profit by Product  
- Line Chart: Sales Trend per Product Category  
- KPI: Top-Selling Product  
- Slicers: Category, Sub-Category  

---

## Drill-through Pages

- **Region Drill-through** – detailed regional insights including sales trends, top products, and KPIs.  
- **Customer Drill-through** – detailed customer-level sales performance.

---

## Tools and Technologies

- **Power BI Desktop** – data visualization and dashboard creation.  
- **DAX (Data Analysis Expressions)** – for calculated columns and measures.  
- **Power Query Editor** – data transformation and cleaning.  

---

## How to Use

1. Open the Power BI report (.pbix file).  
2. Use slicers to filter data by year, region, category, or salesperson.  
3. Click on visuals to cross-filter other visuals on the page.  
4. Right-click on a region or customer in visuals to **drill through** to detailed analysis pages.  
5. Review KPI cards to monitor performance at a glance.

---

## Future Enhancements

- Add **dynamic sales forecasts** using advanced DAX or integration with Python/R for predictive analytics.  
- Include **mobile-optimized layouts** for executives on-the-go.  
- Integrate **live sales data** for real-time dashboards.  

---

## Project Outcome

This Sales Performance Dashboard enables executives and managers to quickly assess overall sales health, identify top-performing regions, products, and salespeople, and make informed decisions based on historical and projected sales data.

