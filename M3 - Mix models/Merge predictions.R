

# -- library
library(ggplot2)
library(gridExtra)

# -- Declare objects
model_1_url <- "predictions_2020_Model_v1.csv"
model_2_url <- "predictions_2020_Model_v2.csv"


# -- Init
# set working directory
setwd("D:/AI/Projects/Australia Rain Tomorrow/Version.3 - Mix models")


# -- Load predictions
predictions_1 <- read.csv(model_1_url, sep = ',', header = TRUE)
predictions_2 <- read.csv(model_2_url, sep = ',', header = TRUE)


# -- Merge dataframes
cols = c("Date", "RainTomorrow", "Prediction")
merged_df <- cbind(predictions_1[cols], predictions_2$Prediction)
cols = c("Date", "RainTomorrow", "Prediction_v1", "Prediction_v2")
colnames(merged_df) <- cols


# -- Replace Yes/No by 1/0
merged_df$RainTomorrow <- ifelse(merged_df$RainTomorrow == "Yes", 1, 0)

# -- Add accuracy columns (floor(0.5+x) to avoid 0.5 to become 0!)
merged_df$RawAccuracy_P1 <- ifelse(merged_df$RainTomorrow == floor(0.5 + merged_df$Prediction_v1),1,0)
merged_df$RawAccuracy_P2 <- ifelse(merged_df$RainTomorrow == floor(0.5 + merged_df$Prediction_v2),1,0)

# -- Compute stats
nb_prediction <- dim(merged_df)[1]
nb_rain_tomorrow <- sum(merged_df$RainTomorrow)

nb_prediction_ok_P1 <- sum(merged_df$RawAccuracy_P1)
nb_prediction_ok_P2 <- sum(merged_df$RawAccuracy_P2)

accuracy_P1 <- nb_prediction_ok_P1 / nb_prediction
accuracy_P2 <- nb_prediction_ok_P2 / nb_prediction

nb_true_positive_P1 <- sum(ifelse((floor(0.5 + merged_df$Prediction_v1) == 1) & (merged_df$RainTomorrow == 1),1,0))
nb_false_positive_P1 <- sum(ifelse((floor(0.5 + merged_df$Prediction_v1) == 1) & (merged_df$RainTomorrow == 0),1,0))
nb_true_negative_P1 <- sum(ifelse((floor(0.5 + merged_df$Prediction_v1) == 0) & (merged_df$RainTomorrow == 0),1,0))
nb_false_negative_P1 <- sum(ifelse((floor(0.5 + merged_df$Prediction_v1) == 0) & (merged_df$RainTomorrow == 1),1,0))

nb_true_positive_P2 <- sum(ifelse((floor(0.5 + merged_df$Prediction_v2) == 1) & (merged_df$RainTomorrow == 1),1,0))


# -- Compute composite predictions
steps <- seq(0, 1, by = 0.001)

# -- Optimize composite model
i <- 1
ref_accuracy <- 0
ref_precision <- 0
ref_recall <- 0

tmp_df <- merged_df$Date

for (a in steps)
{
  # update b
  b <- 1 - a
  
  # compute composite prediction
  tmp_df$Composite <- a * merged_df$Prediction_v1 + b * merged_df$Prediction_v2
  
  # compute accuracy
  tmp_df$RawAccuracy <- ifelse(merged_df$RainTomorrow == floor(0.5 + tmp_df$Composite),1,0)
  
  # compute stats
  tmp_nb_prediction_ok <- sum(tmp_df$RawAccuracy)
  tmp_accuracy <- tmp_nb_prediction_ok / nb_prediction
  tmp_nb_true_positive <- sum(ifelse((floor(0.5 + tmp_df$Composite) == 1) & (merged_df$RainTomorrow == 1),1,0))
  tmp_nb_false_positive <- sum(ifelse((floor(0.5 + tmp_df$Composite) == 1) & (merged_df$RainTomorrow == 0),1,0))
  tmp_nb_true_negative <- sum(ifelse((floor(0.5 + tmp_df$Composite) == 0) & (merged_df$RainTomorrow == 0),1,0))
  tmp_nb_false_negative <- sum(ifelse((floor(0.5 + tmp_df$Composite) == 0) & (merged_df$RainTomorrow == 1),1,0))
  # Precision, recall
  tmp_precision <- tmp_nb_true_positive / (tmp_nb_true_positive + tmp_nb_false_positive)
  tmp_recall <- tmp_nb_true_positive / (tmp_nb_true_positive + tmp_nb_false_negative)
  #F1 Score
  f1_score = (2 * tmp_precision * tmp_recall) / (tmp_precision + tmp_recall)
  
  # save best accuracy
  if (tmp_accuracy > ref_accuracy)
  {
    ref_accuracy <- tmp_accuracy
    
    opt_a_accuracy <- a
    opt_b_accuracy <- b
    
    merged_df$Composite_OptAcc <- tmp_df$Composite
    merged_df$RawAccuracy_OptAcc <- tmp_df$RawAccuracy
    
    nb_prediction_ok_OptAcc <- tmp_nb_prediction_ok
    accuracy_OptAcc <- tmp_accuracy
    nb_true_positive_OptAcc <- tmp_nb_true_positive
    nb_false_positive_OptAcc <- tmp_nb_false_positive
    nb_true_negative_OptAcc <- tmp_nb_true_negative
    nb_false_negative_OptAcc <- tmp_nb_false_negative
  }
  
  # save best precision
  if (tmp_precision > ref_precision)
  {
    ref_precision <- tmp_precision
    
    opt_a_precision <- a
    opt_b_precision <- b
    
    merged_df$Composite_OptPre <- tmp_df$Composite
    merged_df$RawAccuracy_OptPre <- tmp_df$RawAccuracy
    
    nb_prediction_ok_OptPre <- tmp_nb_prediction_ok
    accuracy_OptPre <- tmp_accuracy
    nb_true_positive_OptPre <- tmp_nb_true_positive
    nb_false_positive_OptPre <- tmp_nb_false_positive
    nb_true_negative_OptPre <- tmp_nb_true_negative
    nb_false_negative_OptPre <- tmp_nb_false_negative
  }
  
  # save best recall
  if (tmp_recall > ref_recall)
  {
    ref_recall <- tmp_recall
    
    opt_a_recall <- a
    opt_b_recall <- b
    
    merged_df$Composite_OptRec <- tmp_df$Composite
    merged_df$RawAccuracy_OptRec <- tmp_df$RawAccuracy
    
    nb_prediction_ok_OptRec <- tmp_nb_prediction_ok
    accuracy_OptRec <- tmp_accuracy
    nb_true_positive_OptRec <- tmp_nb_true_positive
    nb_false_positive_OptRec <- tmp_nb_false_positive
    nb_true_negative_OptRec <- tmp_nb_true_negative
    nb_false_negative_OptRec <- tmp_nb_false_negative
  }
  
  i <- i + 1
  
}

# -- Cleaning
rm(a, b, i, steps, tmp_df, tmp_nb_prediction_ok, tmp_accuracy, tmp_precision, tmp_recall,
   tmp_nb_true_positive, tmp_nb_false_positive, tmp_nb_true_negative, tmp_nb_false_negative)



# -- Plot: compare predictions
# Plot.1: distribution of Prediction
p1 <- ggplot(merged_df, aes(x=Optimized_1)) +
  geom_density(alpha=0.4, fill="#E69F00", size=1) +
  geom_vline(aes(xintercept=median(Optimized_1)), color="grey", linetype="dashed", size=1) +
  geom_rug()


# Plot.2: ability to predict days without rain
p2 <- ggplot(merged_df[merged_df$RainTomorrow == 0,], aes( x = Optimized_1)) +
  geom_density(alpha = 0.4, fill = "#E69F00", size = 1) +
  geom_vline(aes(xintercept = median(Optimized_1)), color = "grey", linetype = "dashed", size=1) +
  geom_rug() +
  xlab("Predictions for days without rain")


#Plot.3: ability to predict days with rain
p3 <- ggplot(merged_df[merged_df$RainTomorrow == 1,], aes( x = Optimized_1)) +
  geom_density(alpha = 0.4, fill = "#E69F00", size = 1) +
  geom_vline(aes(xintercept = median(Optimized_1)), color = "grey", linetype = "dashed", size=1) +
  geom_rug() +
  ylim(0,4) +
  xlab("Predictions for days with rain")


# -- Print plots
grid.arrange(p1, p2, p3, ncol = 1, nrow = 3)