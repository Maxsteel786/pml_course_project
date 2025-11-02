##############################################
# Practical Machine Learning - Course Project
# Author: Ayan Mohamed
# Description: Predicting exercise manner using accelerometer data
##############################################

# Load necessary libraries
library(caret)
library(randomForest)
library(dplyr)

# Set seed for reproducibility
set.seed(1234)

# Step 1: Load data
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

train_data <- read.csv(train_url, na.strings = c("NA", "#DIV/0!", ""))
test_data  <- read.csv(test_url,  na.strings = c("NA", "#DIV/0!", ""))

# Step 2: Remove ID/time/window columns if they exist
cols_to_remove <- c("X", "X1", "user_name", "raw_timestamp_part_1", 
                    "raw_timestamp_part_2", "cvtd_timestamp", 
                    "new_window", "num_window")

# Remove only existing columns safely
cols_to_remove <- intersect(cols_to_remove, names(train_data))

train_data <- train_data %>%
  select(-all_of(cols_to_remove))

test_data <- test_data %>%
  select(-all_of(intersect(cols_to_remove, names(test_data))))

# Step 3: Keep only columns with no missing values
train_data <- train_data %>% select_if(~ sum(is.na(.)) == 0)
test_data  <- test_data %>% select_if(~ sum(is.na(.)) == 0)

# Step 4: Split training dataset
inTrain <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
training <- train_data[inTrain, ]
testing  <- train_data[-inTrain, ]

# Step 5: Train Random Forest model
control <- trainControl(method = "cv", number = 3)
rf_model <- train(classe ~ ., data = training,
                  method = "rf",
                  trControl = control,
                  ntree = 100)

# Step 6: Evaluate on validation set
predictions <- predict(rf_model, testing)
conf_mat <- confusionMatrix(predictions, testing$classe)

cat("\n=== Validation Results ===\n")
print(conf_mat)
cat("\nAccuracy:", round(conf_mat$overall['Accuracy'], 4), "\n")

# Step 7: Train final model on full training data
final_model <- randomForest(classe ~ ., data = train_data, ntree = 200)

# Step 8: Predict 20 test cases
final_predictions <- predict(final_model, test_data)

# Step 9: Save predictions
write.table(final_predictions, file = "pml_predictions.txt",
            row.names = FALSE, col.names = FALSE, quote = FALSE)

cat("\n✅ Predictions saved successfully to 'pml_predictions.txt'\n")
##############################################


