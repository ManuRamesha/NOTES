### 1. **Batch and Online Learning**
   - **Batch Learning:** In batch learning, the model is trained on the entire dataset at once. This means the model requires all data to be available beforehand. It's computationally expensive but provides a more accurate model because the system has access to all the data.
     - **When to use:** When the dataset is small, and it's possible to compute all data in a single training cycle.
     - **How it works:** It processes the whole dataset in one go, and after training, the model is deployed. It does not update with new data until the next training cycle.

   - **Online Learning:** Online learning is a more incremental approach where the model learns one example at a time, updating the model with every new data point. This is useful when the data is continuously arriving (such as in real-time systems).
     - **When to use:** In streaming data scenarios or when working with large datasets that don't fit in memory.
     - **How it works:** The model adjusts weights as each data point arrives, thus making predictions in real time.

### 2. **Instance-Based vs. Model-Based Learning**
   - **Instance-Based Learning (IBL):** In IBL, the model memorizes the training instances and makes predictions by comparing the current input with those instances (e.g., k-Nearest Neighbors, k-NN).
     - **When to use:** When you need to handle complex decision boundaries, and real-time updates are required.
     - **How it works:** The model does not generalize but relies on training data directly to make decisions based on the nearest instances.

   - **Model-Based Learning:** This approach builds an abstract model that generalizes from the training data (e.g., Linear Regression, Decision Trees). The model generalizes to unseen data based on the learned patterns.
     - **When to use:** When the goal is to make predictions based on learned patterns.
     - **How it works:** The model learns a mathematical representation of the data, which can be used to predict unseen data points.

### 3. **Data Cleaning**
   - **What it is:** Data cleaning is the process of identifying and rectifying errors or inconsistencies in a dataset. This includes handling missing values, removing duplicates, correcting typos, and standardizing formats.
   - **Why it's used:** To ensure that the data is accurate, complete, and consistent before it’s used in model training. Clean data leads to better model performance.
   - **How it works:** Techniques include:
     - Removing or imputing missing values (null handling).
     - Fixing inconsistent data types.
     - Handling outliers.
     - Addressing duplicate entries.

### 4. **What Are Outliers?**
   - **What they are:** Outliers are data points that significantly deviate from the rest of the data, lying far away from other observations.
   - **Why they matter:** Outliers can distort statistical analyses and impact the accuracy of machine learning models, often leading to incorrect predictions.
   - **How they are detected:** Outliers can be identified using statistical methods such as:
     - Z-scores or IQR (Interquartile Range).
     - Visualization methods like box plots and scatter plots.

### 5. **Filling Null Values (Mean, Median, Mode)**
   - **When to use:** When there are missing or null values in the dataset that must be filled in order for the model to process the data.
   - **Why these are used:** 
     - **Mean**: Used when the data is normally distributed or continuous and not affected by outliers.
     - **Median**: Preferred when the data is skewed or contains outliers, as the median is less sensitive to extreme values.
     - **Mode**: Used for categorical data where you replace missing values with the most frequent category.
   - **How it works:** The null values are replaced by the chosen value (mean, median, or mode), ensuring no loss of data due to missing values.

### 6. **Duplication**
   - **What it is:** Duplication refers to repeated rows in the dataset, which do not add new information.
   - **Why it matters:** Duplicates can introduce bias and lead to overfitting, as the model may place undue importance on repeated instances.
   - **How it is handled:** Duplicates can be identified using data deduplication techniques (e.g., using the `.drop_duplicates()` function in pandas) and removed to ensure each instance is unique.

### 7. **Skewness (Types and Handling)**
   - **What it is:** Skewness is a measure of the asymmetry of the distribution of data. Positive skew means the right tail is longer, while negative skew means the left tail is longer.
   - **Types:**
     - **Positive Skew:** Most values are on the left, but the tail is on the right (e.g., income distribution).
     - **Negative Skew:** Most values are on the right, but the tail is on the left (e.g., age at death).
   - **How to handle skewness:**
     - **Logarithmic Transformation:** Used for highly skewed data with large outliers, this compresses the range of values.
     - **Square Root Transformation:** Good for moderately skewed data.
     - **Quantile Transformation:** Maps data to a uniform or normal distribution, useful for extreme skewness.

### 8. **Correlation**
   - **What it is:** Correlation measures the relationship between two or more variables. It tells you how strongly two variables are related.
   - **Why it matters:** Understanding correlations helps identify patterns, redundancy, and multicollinearity in features.
   - **How it's measured:** 
     - Pearson Correlation: Measures linear relationship.
     - Spearman Correlation: Measures monotonic relationship.
     - Kendall Tau: Measures ordinal association.

### 9. **fit(), transform(), and fit_transform()**
   - **fit():** This method is used to compute and learn the parameters of a model or transformer (e.g., learning the mean and standard deviation for standardization).
   - **transform():** This method applies the learned transformation to the data (e.g., scaling data after fitting).
   - **fit_transform():** A combination of both `fit()` and `transform()`, which first fits the model to the data and then transforms the data using the fitted parameters.

### 10. **PCA (Principal Component Analysis)**
   - **What it is:** PCA is a dimensionality reduction technique that transforms the data into a new coordinate system, selecting the axes (principal components) that explain the most variance in the data.
   - **Why it’s used:** To reduce the number of features (dimensionality) while retaining as much variance as possible.
   - **How it works:** PCA identifies orthogonal axes (principal components) that capture the maximum variance in the data, enabling dimensionality reduction while preserving information.

### 11. **Feature Selection Methods**
   - **Information Gain:** Measures how much information a feature contributes to reducing uncertainty (entropy) in a decision tree. Higher information gain means better feature for splitting data.
   - **Chi-Square Test:** Used to assess the independence of categorical features. A low p-value indicates that a feature is statistically significant for prediction.
   - **Fisher’s Score:** A statistical measure that ranks features based on their ability to discriminate between different classes.
   - **Correlation Coefficient:** Measures the linear relationship between two continuous features. Highly correlated features can be dropped to avoid multicollinearity.
   - **Variance Threshold:** Features with very low variance are likely to be uninformative and can be discarded.
   - **Mean Absolute Difference (MAD):** Measures the variability of a feature and can be used for feature selection in time series or when identifying stable features.
   - **Forward Feature Selection:** Starts with no features and iteratively adds the feature that most improves model performance.
   - **Backward Feature Elimination:** Starts with all features and iteratively removes the least important features.
   - **Exhaustive Feature Selection:** Evaluates all possible combinations of features, selecting the best performing set.
   - **Recursive Feature Elimination (RFE):** Recursively removes the least important features, ranking them by model performance.
   - **LASSO (L1 Regularization):** Encourages sparse solutions by adding a penalty to the absolute value of the coefficients, which forces some feature coefficients to zero, effectively removing them.
   - **Random Forest Importance:** Measures the importance of features by evaluating the average decrease in node impurity when a feature is used to split the data.

### **Best Feature Selection Methods for Skewed and Outlier-Dominated Datasets:**
   - **Log Transformations:** Use them to reduce skewness in features before applying feature selection techniques.
   - **Random Forest Feature Importance:** A robust method that handles outliers well, as tree-based methods are less sensitive to extreme values.
   - **Recursive Feature Elimination (RFE):** Can be effective because it iteratively refines feature selection.
   - **LASSO Regularization:** Works well when there are many features and helps to remove irrelevant ones, especially in the presence of outliers and skewed data.
