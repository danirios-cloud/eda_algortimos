---
author: "Daniela Rios"
output:
  html_document:
    mathjax: true
    keep_md: true
    highlight: zenburn
    theme:  spacelab
  pdf_document:
always_allow_html: true
---




****

<span><h1 style = "font-family: verdana; font-size: 26px; font-style: normal; letter-spcaing: 3px; background-color: #ffe5d9; color :#000000; border-radius: 100px 100px; text-align:center"> 🧠 Machine Learning Pipeline for Stroke Prediction </h1></span>

<b><span style='color:#E888BB; font-size: 16px;'> |</span> <span style='color:#000;'>Evaluation of Traditional Machine Learning Algorithms for Stroke Prediction</span> </b>

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Introduction</strong></div>
<p style='color:#000;'>This pipeline aims to predict stroke occurrences based on patient data. The dataset includes several health-related attributes such as age, gender, glucose levels, and body mass index (BMI). By analyzing these factors, the models provide insights into which patients might be at higher risk of stroke, allowing for earlier interventions and better management.</p>

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Data Loading and Cleaning</strong></div>
<p style='color:#000;'>The data is loaded and cleaned to prepare it for machine learning. This includes handling missing values, converting categorical variables to appropriate data types, and scaling numerical features. Median imputation is used to handle missing values in numerical columns, which is an effective way to retain data without introducing bias.</p>


```r
# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)
library(ROSE)
library(mice)
library(DMwR2)
library(knitr)

# Load the dataset
stroke_data <- read.csv("stroke.csv")

# Inspect the data
str(stroke_data)
```

```
'data.frame':	5110 obs. of  12 variables:
 $ id               : int  9046 51676 31112 60182 1665 56669 53882 10434 27419 60491 ...
 $ gender           : chr  "Male" "Female" "Male" "Female" ...
 $ age              : num  67 61 80 49 79 81 74 69 59 78 ...
 $ hypertension     : int  0 0 0 0 1 0 1 0 0 0 ...
 $ heart_disease    : int  1 0 1 0 0 0 1 0 0 0 ...
 $ ever_married     : chr  "Yes" "Yes" "Yes" "Yes" ...
 $ work_type        : chr  "Private" "Self-employed" "Private" "Private" ...
 $ Residence_type   : chr  "Urban" "Rural" "Rural" "Urban" ...
 $ avg_glucose_level: num  229 202 106 171 174 ...
 $ bmi              : chr  "36.6" "N/A" "32.5" "34.4" ...
 $ smoking_status   : chr  "formerly smoked" "never smoked" "never smoked" "smokes" ...
 $ stroke           : int  1 1 1 1 1 1 1 1 1 1 ...
```

```r
sum(is.na(stroke_data)) # Check for missing values
```

```
[1] 0
```

```r
# Preprocess the data
# Handling missing values using Median Imputation
numerical_columns <- c("age", "avg_glucose_level", "bmi")

# Convert 'bmi' to numeric, handling "N/A" values
stroke_data$bmi <- as.numeric(replace(stroke_data$bmi, stroke_data$bmi == "N/A", NA))

# Remove rows with any remaining non-numeric or infinite values in numerical columns
stroke_data <- stroke_data %>% filter(if_all(all_of(numerical_columns), ~ !is.na(.) & is.finite(.)))

# Check if numerical columns are empty after filtering
if (nrow(stroke_data) == 0) {
  stop("Numerical columns are empty after filtering. Please check your data.")
}

for (col in numerical_columns) {
  if (sum(is.na(stroke_data[[col]])) > 0) {
    stroke_data[[col]][is.na(stroke_data[[col]])] <- median(stroke_data[[col]], na.rm = TRUE)
  }
}

# Convert categorical variables to factors
stroke_data$gender <- as.factor(stroke_data$gender)
stroke_data$ever_married <- as.factor(stroke_data$ever_married)
stroke_data$work_type <- as.factor(stroke_data$work_type)
stroke_data$Residence_type <- as.factor(stroke_data$Residence_type)
stroke_data$smoking_status <- as.factor(stroke_data$smoking_status)
stroke_data$stroke <- as.factor(stroke_data$stroke)

# Apply Min-Max Scaling to numerical features
preproc <- preProcess(stroke_data[, numerical_columns], method = c("range"))
stroke_data[, numerical_columns] <- predict(preproc, stroke_data[, numerical_columns])

# Apply one-hot encoding for categorical variables
# Remove categorical columns with only one level to avoid errors in dummy encoding
categorical_columns <- c("gender", "ever_married", "work_type", "Residence_type", "smoking_status")
categorical_columns <- categorical_columns[sapply(stroke_data[, categorical_columns], function(x) length(unique(x))) > 1]

if (length(categorical_columns) > 0) {
  dummy_model <- dummyVars(~ ., data = stroke_data[, categorical_columns], fullRank = TRUE)
  dummy_encoded <- predict(dummy_model, newdata = stroke_data) %>% as.data.frame()
  
  # Combine encoded categorical columns with the rest of the dataset
  stroke_data <- cbind(stroke_data %>% select(-all_of(categorical_columns)), dummy_encoded)
  colnames(stroke_data) <- make.names(colnames(stroke_data), unique = TRUE)
} else {
  warning("No categorical columns to encode.")
}

# Remove highly correlated features to avoid multicollinearity
cor_matrix <- cor(stroke_data %>% select(all_of(numerical_columns)))
highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.9)
if (length(highly_correlated) > 0) {
  stroke_data <- stroke_data %>% select(-all_of(names(stroke_data)[highly_correlated]))
}
```

<p style='color:#000;'>**Handling Imbalanced Data**: In this part, it was crucial to use upsampling to balance the dataset, as the number of stroke cases was significantly lower than non-stroke cases. This ensures that the model does not become biased towards predicting the majority class, thus improving the model's sensitivity to stroke cases.</p>


```r
# Handling imbalanced data using upSample from caret with lower duplication
stroke_data$stroke <- factor(stroke_data$stroke, levels = c(0, 1))
stroke_data_balanced <- upSample(x = stroke_data %>% select(-stroke), y = stroke_data$stroke, yname = "stroke")
```

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Splitting the Dataset</strong></div>
<p style='color:#000;'>The balanced dataset is split into training, testing, and validation sets in a 70-20-10 ratio. This division ensures that the model is trained on a significant portion of the data while maintaining separate sets for unbiased evaluation.</p>


```r
# Split the dataset into training, testing, and validation sets (70-20-10 split)
set.seed(123)
train_index <- createDataPartition(stroke_data_balanced$stroke, p = 0.7, list = FALSE)
train_data <- stroke_data_balanced[train_index, ]
temp_data <- stroke_data_balanced[-train_index, ]

# Further split temp_data into testing (20%) and validation (10%)
test_index <- createDataPartition(temp_data$stroke, p = 2/3, list = FALSE)
test_data <- temp_data[test_index, ]
validation_data <- temp_data[-test_index, ]
```

<div style="background-color:#b2f7ef; color:black; border-radius:5px; padding: 0.2em 0.5em; display: inline-block;"><strong>Model Training</strong></div>
<p style='color:#000;'>The models used in this pipeline do not require a normal distribution of the features. Models such as decision trees, random forests, and XGBoost are inherently non-parametric and robust to outliers. Hence, strict normality checks or transformations were not necessary.</p>


```r
# Train Logistic Regression model
logistic_model <- train(stroke ~ ., data = train_data, method = "glm", family = "binomial")
logistic_pred <- predict(logistic_model, newdata = test_data)
logistic_cm <- confusionMatrix(logistic_pred, test_data$stroke)
cat("\n\033[1mLogistic Regression Confusion Matrix:\033[0m\n")
```

```

[1mLogistic Regression Confusion Matrix:[0m
```

```r
kable(as.data.frame(logistic_cm$table), format = "markdown", caption = "Logistic Regression Confusion Matrix")
```



Table: Logistic Regression Confusion Matrix

|Prediction |Reference | Freq|
|:----------|:---------|----:|
|0          |0         |  707|
|1          |0         |  233|
|0          |1         |  191|
|1          |1         |  749|

```r
# Train Decision Tree model
dtree_model <- train(stroke ~ ., data = train_data, method = "rpart")
dtree_pred <- predict(dtree_model, newdata = test_data)
dtree_cm <- confusionMatrix(dtree_pred, test_data$stroke)
cat("\n\033[1mDecision Tree Confusion Matrix:\033[0m\n")
```

```

[1mDecision Tree Confusion Matrix:[0m
```

```r
kable(as.data.frame(dtree_cm$table), format = "markdown", caption = "Decision Tree Confusion Matrix")
```



Table: Decision Tree Confusion Matrix

|Prediction |Reference | Freq|
|:----------|:---------|----:|
|0          |0         |  602|
|1          |0         |  338|
|0          |1         |   92|
|1          |1         |  848|

```r
# Train Random Forest model (Adjusted Complexity)
rf_model <- randomForest(
  stroke ~ ., 
  data = train_data, 
  ntree = 80,  # Increase number of trees slightly to capture more patterns
  maxnodes = 15  # Allow more nodes to improve learning capacity while still limiting complexity
)
rf_pred <- predict(rf_model, newdata = test_data)
rf_cm <- confusionMatrix(rf_pred, test_data$stroke)
cat("\n\033[1mRandom Forest Confusion Matrix (Adjusted Complexity):\033[0m\n")
```

```

[1mRandom Forest Confusion Matrix (Adjusted Complexity):[0m
```

```r
kable(as.data.frame(rf_cm$table), format = "markdown", caption = "Random Forest Confusion Matrix (Adjusted Complexity)")
```



Table: Random Forest Confusion Matrix (Adjusted Complexity)

|Prediction |Reference | Freq|
|:----------|:---------|----:|
|0          |0         |  659|
|1          |0         |  281|
|0          |1         |  115|
|1          |1         |  825|


```r
# Train XGBoost model
# Convert data to matrix format for XGBoost
train_matrix <- model.matrix(stroke ~ . - 1, data = train_data)
train_label <- as.numeric(train_data$stroke) - 1
test_matrix <- model.matrix(stroke ~ . - 1, data = test_data)
test_label <- as.numeric(test_data$stroke) - 1

# Define DMatrix objects for training and testing
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# Define and train the XGBoost model with Reduced Complexity
xgb_model <- xgboost(
  data = dtrain, 
  nrounds = 50,  # Reduced number of boosting rounds
  objective = "binary:logistic", 
  eta = 0.1,  # Learning rate to prevent overfitting
  max_depth = 4,  # Reduced maximum depth to limit model complexity
  verbose = 0
)

# Make predictions and evaluate XGBoost model
xgb_pred_prob <- predict(xgb_model, newdata = dtest)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0)
xgb_cm <- confusionMatrix(as.factor(xgb_pred), as.factor(test_label))

# Print the Confusion Matrix
cat("\n\033[1mXGBoost Confusion Matrix (Reduced Complexity):\033[0m\n")
```

```

[1mXGBoost Confusion Matrix (Reduced Complexity):[0m
```


```r
# Complete the model comparison data frame
model_comparison <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"),
  Accuracy = c(logistic_cm$overall["Accuracy"],
               dtree_cm$overall["Accuracy"],
               rf_cm$overall["Accuracy"],
               xgb_cm$overall["Accuracy"]),
  Sensitivity = c(logistic_cm$byClass["Sensitivity"],
                  dtree_cm$byClass["Sensitivity"],
                  rf_cm$byClass["Sensitivity"],
                  xgb_cm$byClass["Sensitivity"]),
  Specificity = c(logistic_cm$byClass["Specificity"],
                  dtree_cm$byClass["Specificity"],
                  rf_cm$byClass["Specificity"],
                  xgb_cm$byClass["Specificity"])
)

print(model_comparison)
```

```
                Model  Accuracy Sensitivity Specificity
1 Logistic Regression 0.7744681   0.7521277   0.7968085
2       Decision Tree 0.7712766   0.6404255   0.9021277
3       Random Forest 0.7893617   0.7010638   0.8776596
4             XGBoost 0.8632979   0.7936170   0.9329787
```



```r
# Plot ROC Curves for all models
# Logistic Regression ROC
logistic_roc <- roc(test_data$stroke, as.numeric(predict(logistic_model, test_data, type = "prob")[, 2]))

# Decision Tree ROC
dtree_roc <- roc(test_data$stroke, as.numeric(predict(dtree_model, test_data, type = "prob")[, 2]))

# Random Forest ROC
rf_roc <- roc(test_data$stroke, as.numeric(predict(rf_model, test_data, type = "prob")[, 2]))

# XGBoost ROC
xgb_roc <- roc(test_label, xgb_pred_prob)

# Plot all ROC curves
plot(logistic_roc, col = "blue", lwd = 2, main = "ROC Curves for Stroke Prediction Models")
plot(dtree_roc, col = "green", lwd = 2, add = TRUE)
plot(rf_roc, col = "red", lwd = 2, add = TRUE)
plot(xgb_roc, col = "purple", lwd = 2, add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"),
       col = c("blue", "green", "red", "purple"), lwd = 2)
```

![](stroke_algoritmos_tradicionales_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
# Display AUC for all models
cat("\nAUC Values:\n")
```

```

AUC Values:
```

```r
cat("Logistic Regression AUC: ", auc(logistic_roc), "\n")
```

```
Logistic Regression AUC:  0.8480953 
```

```r
cat("Decision Tree AUC: ", auc(dtree_roc), "\n")
```

```
Decision Tree AUC:  0.8206881 
```

```r
cat("Random Forest AUC: ", auc(rf_roc), "\n")
```

```
Random Forest AUC:  0.8730461 
```

```r
cat("XGBoost AUC: ", auc(xgb_roc), "\n")
```

```
XGBoost AUC:  0.9228175 
```


<br><br><br><br>
