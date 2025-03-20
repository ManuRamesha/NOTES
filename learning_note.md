# Machine Learning 
## Anamoly detection


# Anomaly Detection for Dataset with 30 Features

## Objective
The aim of this project is to detect anomalies in a dataset containing 30 features where the target variable `Class` indicates whether an observation is an anomaly (`1`) or not (`0`). The dataset is analyzed using both supervised and unsupervised anomaly detection models.

---

## Data Preprocessing

### 1. **Missing Values**
   - **Initial check for null values**: 
     - The dataset was inspected for missing values, and none were found. Therefore, no imputation was necessary.
   - **Handling missing values (if applicable)**:
     - If the dataset had contained missing values, several strategies could be used to handle them:
       - **Mean Imputation**: Suitable for numerical columns that are approximately normally distributed. It replaces missing values with the mean of the available data.
       - **Median Imputation**: Best used for numerical columns with skewed distributions, as the median is less sensitive to outliers compared to the mean.
       - **Mode Imputation**: Used for categorical data to replace missing values with the most frequent value (mode).
       - **Deletion**: In cases where missing data is minimal or not randomly missing, rows or columns can be dropped.
   
### 2. **Duplicate Records**
   - **Duplication check**:
     - The dataset was inspected for duplicate records, and none were found. Therefore, no action was needed for this dataset.
   - **Impact of Duplicates on Anomaly Detection**:
     - **Problem with duplicates**: If duplicates were present in the dataset, it could severely distort anomaly detection results. Anomalies are typically rare and different from the majority of observations. If duplicate records are present, they could be incorrectly classified as "normal" observations, potentially biasing the model and leading to poor anomaly detection performance.
     - **Why duplicates are problematic**:
       - **Data Redundancy**: Duplicates cause unnecessary repetition of the same data points, which might result in misleading conclusions about the distribution and relationships within the dataset.
       - **False Normalization**: In models that rely on distance measures (like DBSCAN, Isolation Forest, or One-Class SVM), duplicates would distort the notion of distance, as they would artificially reduce the variation in the dataset, leading to incorrect clustering or classification.
       - **Threshold Issues**: In supervised models, duplicates could affect threshold determination in classification tasks (e.g., Random Forest), where a high frequency of a particular class due to duplicates may skew the model's decision-making process.
     - **Handling**: Since there were no duplicates in this dataset, there was no need to take corrective action, but it's crucial to always check for duplicates, especially in cases where the dataset might be aggregated from multiple sources.

### 3. **Skewness**
   - **Initial analysis**: 
     - The dataset exhibited both left-skewed and right-skewed distributions across multiple features.
   - **Methods for handling skewness**:
     - **Log Transformation**: A common technique to reduce right skewness. It transforms the feature by applying the logarithm function, making the data distribution more symmetric.
     - **Square Root Transformation**: Applied to data with moderate right skewness. It reduces the effect of large values, but less aggressively than the log transformation.
     - **Box-Cox Transformation**: A more flexible approach that can handle both positive and negative skewness by finding the optimal power transformation to make the data as normal as possible.
     - **Quantile Transformation**: Applied here, as it maps data to a uniform or normal distribution using percentiles. This is especially useful when the data is skewed on both sides.
     - **Yeo-Johnson Transformation**: Similar to Box-Cox but works for both positive and negative values. It can be helpful if the dataset includes negative values.
   - **Outcome**: 
     - **Quantile Transformation** was applied to handle both left and right skewness, effectively normalizing the data for better model performance.

### 4. **Outliers**
   - **Initial analysis of outliers**:
     - Outliers were initially identified and considered for removal.
   - **Effect of removing outliers**:
     - After removing outliers, the dataset became imbalanced, as the anomalies (outliers) are the critical data points in anomaly detection tasks.
   - **Conclusion**: 
     - Retained the outliers as they represent the anomalies in the dataset. Removing them would have caused a loss of valuable information.

---

## Model Selection

### Supervised Models:

1. **Random Forest**
   - **Reason for selection**: Random Forest showed the best performance in detecting anomalies. This is due to its ability to handle both balanced and imbalanced datasets by using impurity-based splitting in decision trees. It can classify anomalies effectively by identifying patterns from labeled data.
   - **Advantages**:
     - Robust to overfitting due to its ensemble nature.
     - Handles imbalanced data well, which is critical in anomaly detection.
     - No need for feature scaling or normalization, making it efficient in real-world scenarios.
   - **Contribution to Analysis**:
     - **Threshold Determination**: Random Forest performs well in anomaly detection because it can set appropriate thresholds based on impurity, which is useful for distinguishing anomalies from normal points in imbalanced datasets. This characteristic helps classify anomalous data more accurately.
     - **Feature Importance**: Random Forest can provide insights into which features contribute most to identifying anomalies, aiding in feature selection for future analysis.

2. **Logistic Regression**
   - **Challenges**:
     - Assumes linear relationships between features and the target variable, which may not hold in this case, leading to suboptimal performance for anomaly detection.
   - **Contribution to Analysis**:
     - Logistic regression performed well after Random Forest due to its simplicity and efficiency. While it assumes linearity, it managed to capture enough meaningful patterns from the dataset to perform better than some other models.
     - This model was found to be less effective than Random Forest, but it provided a good baseline for anomaly detection tasks.

### Unsupervised Models:

1. **One-Class SVM**
   - **Reason for selection**: This model is designed specifically for anomaly detection, aiming to learn a decision function for outlier detection in high-dimensional spaces.
   - **Challenges**: 
     - Initially, One-Class SVM with an **RBF kernel** did not perform as well as expected.
   - **Hyperparameter Tuning**: 
     - After **tuning the kernel** from **RBF (Radial Basis Function)** to **Linear**, the performance of the One-Class SVM improved significantly.
     - The linear kernel allowed for better classification of anomalies in this case, as it was able to draw simpler, more effective decision boundaries compared to the non-linear RBF kernel.
   - **Contribution to Analysis**:
     - **Kernel Adjustment**: Switching to the linear kernel made the model more effective in distinguishing anomalies, as it reduced the complexity of the decision boundary and allowed for better fit to the data.
     - One-Class SVM with a linear kernel was able to improve performance after tuning, showcasing the importance of kernel selection in this model.

2. **Isolation Forest**
   - **Reason for selection**: Effective for anomaly detection by isolating observations that differ significantly from others.
   - **Challenges**: 
     - Struggled with highly imbalanced data and did not perform as well as Random Forest in this specific dataset.
   - **Contribution to Analysis**:
     - Isolation Forest is highly efficient in detecting anomalies based on isolation. It works well with large datasets, and its performance can be improved by adjusting the number of trees and sample size. However, in imbalanced datasets like this one, it may not always capture anomalies accurately.

3. **DBSCAN**
   - **Reason for selection**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) can detect outliers as points that don't belong to any cluster.
   - **Challenges**: 
     - While useful for clustering, DBSCAN struggled with the dataset's imbalance and noise, leading to inconsistent anomaly detection.
   - **Contribution to Analysis**:
     - DBSCAN is a density-based algorithm that can find outliers as points in low-density regions. However, it requires careful tuning of parameters (like `epsilon` and `min_samples`) to work effectively, and its performance can degrade with noisy or imbalanced data.

4. **Autoencoders**
   - **Reason for selection**: Autoencoders are neural networks used for unsupervised learning tasks, particularly for anomaly detection by learning a compressed representation of the input data.
   - **Challenges**: 
     - While they were effective at detecting anomalies, their performance did not match the simplicity and efficiency of Random Forest in this case.
   - **Contribution to Analysis**:
     - Autoencoders provide an unsupervised approach to anomaly detection by learning the normal data distribution. When an anomaly is introduced, the reconstruction error increases, signaling an outlier. However, they can be computationally expensive and require large datasets for training.

---

## Conclusion

- **Random Forest** outperformed other models in anomaly detection due to its robust handling of imbalanced data and its use of impurity-based classification.
- **Logistic Regression** performed well after Random Forest, offering a simpler alternative that captured enough of the patterns to be useful.
- **One-Class SVM** with a **Linear kernel** showed improved performance after hyperparameter tuning, highlighting the importance of kernel selection in this model.
- **Unsupervised models** such as Isolation Forest, One-Class SVM, and Autoencoders also detected anomalies but were less effective compared to Random Forest in this scenario.
- The key takeaway is that **supervised models**, especially Random Forest, are more reliable when the dataset has labeled data and the goal is to classify anomalies accurately.



