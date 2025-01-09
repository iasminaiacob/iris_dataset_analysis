library(ggplot2)
library(dplyr)
library(caret)

data("iris")
df <- iris


#description
str(df)
summary(df)

#statistics
summary_stats <- sapply(df[, 1:4], function(x) c(mean = mean(x), sd = sd(x), median = median(x), min = min(x), max = max(x)))


#visualizations
#histogram of Sepal.Length
ggplot(df, aes(x = Sepal.Length)) + 
  geom_histogram(binwidth = 0.2, fill = "blue", color = "black") +
  ggtitle("Histogram of Sepal Length")

#boxplot of Sepal.Length by Species
ggplot(df, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_boxplot() +
  ggtitle("Boxplot of Sepal Length by Species")

#scatterplot of Petal.Length vs Petal.Width
ggplot(df, aes(x = Petal.Length, y = Petal.Width, color = Species)) +
  geom_point(size = 3) +
  ggtitle("Scatterplot of Petal Length vs Petal Width by Species")

#hypothesis testing
#ANOVA test
anova_model <- aov(Petal.Length ~ Species, data = df)

#post-hoc test (Tukey's HSD)
tukey_result <- TukeyHSD(anova_model)

#Shapiro-Wilk test for normality
shapiro_setosa <- shapiro.test(df$Petal.Length[df$Species == "setosa"])
shapiro_versicolor <- shapiro.test(df$Petal.Length[df$Species == "versicolor"])
shapiro_virginica <- shapiro.test(df$Petal.Length[df$Species == "virginica"])


#predictive modeling
#data prep
set.seed(123)
trainIndex <- createDataPartition(df$Species, p = 0.7, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

#training
model_knn <- train(Species ~ ., data = train, method = "knn", tuneLength = 5)

#predict on test data
predictions <- predict(model_knn, test)

#confusion matrix
confusion_matrix <- confusionMatrix(predictions, test$Species)


#saving results
results <- list(
  summary_stats = summary_stats,
  anova_result = summary(anova_model),
  tukey_result = tukey_result,
  shapiro_setosa = shapiro_setosa,
  shapiro_versicolor = shapiro_versicolor,
  shapiro_virginica = shapiro_virginica,
  knn_model = model_knn,
  confusion_matrix = confusion_matrix
)
save(results, file = "iris_analysis_results.RData")


#showing the results
load("iris_analysis_results.RData")

#structure of the results object
str(results)

#viewing each result
#summary statistics
results$summary_stats

#ANOVA
results$anova_result

#Tukey's HSD post-hoc test
results$tukey_result

#Shapiro-Wilk normality tests
results$shapiro_setosa
results$shapiro_versicolor
results$shapiro_virginica

#k-NN model summary
results$knn_model

#confusion matrix
results$confusion_matrix
