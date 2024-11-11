# load necessary libraries
library(ggplot2)
library(factoextra)
library(caret) 
library(patchwork)

input_data <- read.csv("hw1_input.csv")
s11_real <- read.csv("hw1_real.csv")
s11_imag <- read.csv("hw1_img.csv")

print("dimension of input data:")
dim(input_data)
print("dimension of S11 real part data:")
dim(s11_real)
print("dimension of S11 imaginary part data:")
dim(s11_imag)

str(input_data)

summary(input_data)

head(input_data)

head(s11_real)

head(s11_imag)

s11_magnitude <- sqrt(s11_real^2 + s11_imag^2)
head(s11_magnitude)

# plot the magnitudes of outputs with respect to frequenciens for the first 10 data:

frequency <- seq(1, ncol(s11_magnitude), length.out = ncol(s11_magnitude))
matplot(frequency, t(s11_magnitude[1:10, ]), type = "l", lty = 1, col = 1:10,
        xlab = "Frequency", ylab = "Magnitude", main = "Magnitude vs Frequency for First 10 Samples")
legend("bottomleft", legend = paste("Sample", 1:10), col = 1:10, lty = 1)


# apply pca for s11 magnitude
s11_magnitude_pca <- prcomp(s11_magnitude, scale = TRUE)
summary(s11_magnitude_pca)  


# scree plot of variance
fviz_eig(s11_magnitude_pca,
        addlabels=TRUE)

# biplot with labeled variables
fviz_pca_biplot(s11_magnitude_pca, 
               label = "var")

# apply pca for s11 real part
s11_real_pca <- prcomp(s11_real, scale = TRUE)
summary(s11_real_pca)  

# scree plot of variance
fviz_eig(s11_real_pca,
        addlabels=TRUE)

# biplot with labeled variables
fviz_pca_biplot(s11_real_pca, 
               label = "var")

# apply pca for s11 imaginary part
s11_imag_pca <- prcomp(s11_imag, scale = TRUE)
summary(s11_imag_pca)  

# scree plot of variance
fviz_eig(s11_imag_pca,
        addlabels=TRUE)

# biplot with labeled variables
fviz_pca_biplot(s11_imag_pca, 
               label = "var")

input_data_pca <- prcomp(input_data, scale = TRUE)
summary(input_data_pca) 


# scree plot of variance
fviz_eig(input_data_pca,
        addlabels=TRUE)

# biplot with default settings
fviz_pca_biplot(input_data_pca,
               label = "var")


# extract scores from the PCA results
input_scores <- input_data_pca$x  
output_scores_real <- s11_real_pca$x 

# calculate correlations between all pairs of PCs
num_input_pcs <- ncol(input_scores)
num_output_pcs <- ncol(output_scores_real)
cor_matrix <- matrix(NA, nrow = num_input_pcs, ncol = num_output_pcs,
                     dimnames = list(paste0("Input_PC", 1:num_input_pcs),
                                     paste0("Real_Output_PC", 1:num_output_pcs)))

# fill the matrix with correlations
for (i in 1:num_input_pcs) {
  for (j in 1:num_output_pcs) {
    cor_matrix[i, j] <- cor(input_scores[, i], output_scores_real[, j])
  }
}

# pairs with high correlations treshold = 0.8
high_cor_pairs <- which(abs(cor_matrix) > 0.8, arr.ind = TRUE)

# display high-correlation pairs with their correlation values
if (nrow(high_cor_pairs) > 0) {
  for (k in 1:nrow(high_cor_pairs)) {
    input_pc <- high_cor_pairs[k, 1]
    output_pc <- high_cor_pairs[k, 2]
    correlation_value <- cor_matrix[input_pc, output_pc]
    cat(sprintf("Input_PC%d and Real_Output_PC%d have a high correlation of %.2f\n",
                input_pc, output_pc, correlation_value))
  }
} else {
  cat("No pairs with high correlation found.\n")
}


# extract scores from the PCA results
input_scores <- input_data_pca$x  
output_scores_imag <- s11_imag_pca$x 

# calculate correlations between all pairs of PCs
num_input_pcs <- ncol(input_scores)
num_output_pcs <- ncol(output_scores_imag)
cor_matrix <- matrix(NA, nrow = num_input_pcs, ncol = num_output_pcs,
                     dimnames = list(paste0("Input_PC", 1:num_input_pcs),
                                     paste0("Imaginary_Output_PC", 1:num_output_pcs)))

# fill the matrix with correlations
for (i in 1:num_input_pcs) {
  for (j in 1:num_output_pcs) {
    cor_matrix[i, j] <- cor(input_scores[, i], output_scores_imag[, j])
  }
}

# pairs with high correlations treshold = 0.5
high_cor_pairs <- which(abs(cor_matrix) > 0.5, arr.ind = TRUE)

# display high-correlation pairs with their correlation values
if (nrow(high_cor_pairs) > 0) {
  for (k in 1:nrow(high_cor_pairs)) {
    input_pc <- high_cor_pairs[k, 1]
    output_pc <- high_cor_pairs[k, 2]
    correlation_value <- cor_matrix[input_pc, output_pc]
    cat(sprintf("Input_PC%d and Imaginary_Output_PC%d have a high correlation of %.2f\n",
                input_pc, output_pc, correlation_value))
  }
} else {
  cat("No pairs with high correlation found.\n")
}

# extract scores from the PCA results
input_scores <- input_data_pca$x  
output_scores_magnitude <- s11_magnitude_pca$x 

# calculate correlations between all pairs of PCs
num_input_pcs <- ncol(input_scores)
num_output_pcs <- ncol(output_scores_magnitude)
cor_matrix <- matrix(NA, nrow = num_input_pcs, ncol = num_output_pcs,
                     dimnames = list(paste0("Input_PC", 1:num_input_pcs),
                                     paste0("Magnitude_Output_PC", 1:num_output_pcs)))

# fill the matrix with correlations
for (i in 1:num_input_pcs) {
  for (j in 1:num_output_pcs) {
    cor_matrix[i, j] <- cor(input_scores[, i], output_scores_magnitude[, j])
  }
}

# pairs with high correlations treshold = 0.3
high_cor_pairs <- which(abs(cor_matrix) > 0.3, arr.ind = TRUE)

# display high-correlation pairs with their correlation values
if (nrow(high_cor_pairs) > 0) {
  for (k in 1:nrow(high_cor_pairs)) {
    input_pc <- high_cor_pairs[k, 1]
    output_pc <- high_cor_pairs[k, 2]
    correlation_value <- cor_matrix[input_pc, output_pc]
    cat(sprintf("Input_PC%d and Magnitude_Output_PC%d have a high correlation of %.2f\n",
                input_pc, output_pc, correlation_value))
  }
} else {
  cat("No pairs with high correlation found.\n")
}

options(repr.plot.width = 18, repr.plot.height = 6)
par(mfrow = c(1, 3))

# Plotting Input_PC1 against Real_Output_PC1
plot(input_scores[, 1], output_scores_real[, 1], 
     xlab = "Input_PC1", ylab = "Real_Output_PC1", 
     main = "Scatter Plot of Input_PC1 vs. Real_Output_PC1")
abline(lm(output_scores_real[, 1] ~ input_scores[, 1]), col = "red") 

# Plotting Input_PC1 against Imaginary_Output_PC1
plot(input_scores[, 1], output_scores_imag[, 1], 
     xlab = "Input_PC1", ylab = "Imaginary_Output_PC1", 
     main = "Scatter Plot of Input_PC1 vs. Imaginary_Output_PC1")
abline(lm(output_scores_imag[, 1] ~ input_scores[, 1]), col = "red") 

# Plotting Input_PC1 against Magnitude_Output_PC1
plot(input_scores[, 1], output_scores_magnitude[, 1], 
     xlab = "Input_PC1", ylab = "Magnitude_Output_PC1", 
     main = "Scatter Plot of Input_PC1 vs. Magnitude_Output_PC1")
abline(lm(output_scores_magnitude[, 1] ~ input_scores[, 1]), col = "red") 


loadings_input <- input_data_pca$rotation

loadings_input_pc1 <- loadings_input[, 1]

sorted_loadings_pc1 <- sort(abs(loadings_input_pc1), decreasing = TRUE)

top_features <- names(sorted_loadings_pc1)[1:5]
print(top_features)




# Find the frequency index where the S11 magnitude is minimum
min_magnitude_indices <- apply(s11_magnitude, 1, which.min)

# Extract the frequencies corresponding to the minimum S11 magnitude for each frequency band
resonance_frequencies <- colnames(s11_magnitude)[min_magnitude_indices]


#print(resonance_frequencies)

# Count the frequency of each resonance frequency
resonance_frequency_count <- table(resonance_frequencies)

#print(resonance_frequency_count)

# Sort the count in descending order and get the top 3
top_3_resonance_frequencies <- sort(resonance_frequency_count, decreasing = TRUE)[1:3]

# Print the top 3 resonance frequencies
print(top_3_resonance_frequencies)

# Plot the distribution of resonance frequencies
barplot(table(resonance_frequencies), 
        main = "Distribution of Resonance Frequencies", 
        xlab = "Frequency Labels", 
        ylab = "Count", 
        col = "skyblue", 
        las = 2)  # Rotate x-axis labels for readability


# target frequency
target_frequency_200 <- s11_real[,"X200"]

# fit a linear regression model
model_real_X200 <- lm(target_frequency_200 ~ ., data = input_data)

summary(model_real_X200)

# target frequency
target_frequency_200_imag <- s11_imag[,"X200"]

# fit a linear regression model
model_imag_X200 <- lm(target_frequency_200_imag ~ ., data = input_data)

summary(model_imag_X200)

# target frequency 
target_frequency_200_magnitude <- s11_magnitude[,"X200"]

# fit a linear regression model
model_magnitude_X200 <- lm(target_frequency_200_magnitude ~ ., data = input_data)

summary(model_magnitude_X200)

# generate predictions for real part of output for frequency X200
predicted_values <- predict(model_real_X200, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_200,
  Predicted = predicted_values
)

# plot Predicted vs Actual
real200 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Real Part at Frequency X200",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()



# generate predictions for imaginary part of output for frequency X200
predicted_values <- predict(model_imag_X200, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_200_imag,
  Predicted = predicted_values
)

# plot Predicted vs Actual
imag200 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Imaginary Part at Frequency X200",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()




# generate predictions for magnitude of output for frequency X200
predicted_values <- predict(model_magnitude_X200, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_200_magnitude,
  Predicted = predicted_values
)

# plot Predicted vs Actual
magnitude200 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Magnitude at Frequency X200",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()

options(repr.plot.width = 18, repr.plot.height = 6)  
real200 + imag200 + magnitude200 + plot_layout(ncol = 3)


# target frequency 
target_frequency_0 <- s11_real[,"X0"]

# fit a linear regression model
model_real_X0 <- lm(target_frequency_0 ~ ., data = input_data)

summary(model_real_X0)

# target frequency 
target_frequency_0_imag <- s11_imag[,"X0"]

# fit a linear regression model
model_imag_X0 <- lm(target_frequency_0_imag ~ ., data = input_data)

summary(model_imag_X0)

# target frequency 
target_frequency_0_magnitude <- s11_magnitude[,"X0"]

# fit a linear regression model
model_magnitude_X0 <- lm(target_frequency_0_magnitude ~ ., data = input_data)


summary(model_magnitude_X0)

# generate predictions for real part of output for frequency X200
predicted_values <- predict(model_real_X0, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_0,
  Predicted = predicted_values
)

# plot Predicted vs Actual
real0 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Real Part at Frequency X0",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()


# generate predictions for imaginary part of output for frequency X200
predicted_values <- predict(model_imag_X0, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_0_imag,
  Predicted = predicted_values
)

# plot Predicted vs Actual
imag0 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Imaginary Part at Frequency X0",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()


# generate predictions for magnitude of output for frequency X200
predicted_values <- predict(model_magnitude_X0, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_0_magnitude,
  Predicted = predicted_values
)

# plot Predicted vs Actual
magnitude0 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Magnitude at Frequency X0",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()


options(repr.plot.width = 18, repr.plot.height = 6)

real0 + imag0 + magnitude0 + plot_layout(ncol = 3)

# target frequency 
target_frequency_114 <- s11_real[,"X114"]

# fit a linear regression model
model_real_X114 <- lm(target_frequency_114 ~ ., data = input_data)

summary(model_real_X114)

# target frequency 
target_frequency_114_imag <- s11_imag[,"X114"]

# fit a linear regression model
model_imag_X114 <- lm(target_frequency_114_imag ~ ., data = input_data)

summary(model_imag_X114)

# target frequency 
target_frequency_114_magnitude <- s11_magnitude[,"X114"]

# fit a linear regression model
model_magnitude_X114 <- lm(target_frequency_114_magnitude ~ ., data = input_data)

summary(model_magnitude_X114)

# generate predictions for real part of output for frequency X114
predicted_values <- predict(model_real_X114, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_114,
  Predicted = predicted_values
)

# plot Predicted vs Actual
real114 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Real Part at Frequency X114",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()


# generate predictions for imaginary part of output for frequency X114
predicted_values <- predict(model_imag_X114, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_114_imag,
  Predicted = predicted_values
)

# plot Predicted vs Actual
imag114 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Imaginary Part at Frequency X114",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()


# generate predictions for magnitude of output for frequency X114
predicted_values <- predict(model_magnitude_X114, newdata = input_data)

# data frame with actual and predicted values
comparison_data <- data.frame(
  Actual = target_frequency_114_magnitude,
  Predicted = predicted_values
)

# plot Predicted vs Actual
magnitude114 <- ggplot(comparison_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values for S11 Magnitude at Frequency X114",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()

options(repr.plot.width = 18, repr.plot.height = 6) 

real114 + imag114 + magnitude114 + plot_layout(ncol = 3)


