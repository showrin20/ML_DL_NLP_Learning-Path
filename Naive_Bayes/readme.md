

# **Naive Bayes Classifier**
**Definition:** A probabilistic classifier based on Bayes' theorem, assuming conditional independence of features.

**Steps:**
1. **Training Phase:**
   - Calculate prior probability: P(y)
   - Calculate conditional probability for each feature: P(f|y)

2. **Prediction Phase:**
   - For new data, calculate: P(y|f1, f2, ..., fn)
   - Choose the class with the highest probability.

**Formula:**
\[
P(y|X) \propto P(y) \prod_{i} P(f_i|y)
\]

**Example:** For the sentence *"predictable with no fun"*:
- Calculate P(positive) and P(negative) using word counts.
- Multiply probabilities and select the higher value.

**Smoothing:** Avoids zero probabilities by adding a constant (Laplace Smoothing).
\[
P(w_i|c) = \frac{count(w_i, c) + 1}{\sum_{w \in V} count(w, c) + |V|}
\]

**Variants:**
1. **Multinomial Naive Bayes:** Based on word frequency.
2. **Binary Multinomial Naive Bayes:** Only considers word presence (not frequency).

**Strengths:** Simple, efficient, good for high-dimensional data.
**Weaknesses:** Assumes independence, sensitive to feature correlations.

---

### 5. **ML Evaluation**

**1. Data Splitting:**
- **Train set:** Model training.
- **Validation set:** Parameter tuning.
- **Test set:** Final evaluation.
- **Cross-validation:** K-fold method to use all data efficiently.

**2. Performance Metrics:**
- **Accuracy:** \( \frac{TP + TN}{TP + TN + FP + FN} \)
- **Precision:** \( \frac{TP}{TP + FP} \) (How many predicted positives were correct?)
- **Recall (Sensitivity):** \( \frac{TP}{TP + FN} \) (How many actual positives were detected?)
- **F1 Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Table showing TP, FP, TN, FN.

**Example Calculation:**
|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | TP = 50            | FN = 10            |
| **Actual Negative** | FP = 5             | TN = 100           |

- Precision = 50 / (50 + 5) = 0.91
- Recall = 50 / (50 + 10) = 0.83
- F1 Score = 2 × (0.91 × 0.83) / (0.91 + 0.83) = 0.87

**3. Addressing Issues:**
- **Overfitting:** Use regularization, simplify model.
- **Underfitting:** Increase model complexity, collect more data.

**4. Statistical Significance:**
- **Null Hypothesis (H0):** No difference between models.
- **T-test:** Checks if observed improvement is statistically significant.

---

### 6. **Applications and Use Cases**
1. **Text Classification:** Spam detection, sentiment analysis.
2. **Sentiment Analysis:** Extracting positive or negative opinions.
3. **Language Identification:** Determining language of a text.
4. **Authorship Attribution:** Identifying the author based on writing style.

---

### 7. **Key Takeaways**
- Understand feature representation methods (BoW, N-grams, Lexicons).
- Master conditional probability and Bayes' theorem.
- Be proficient with Naive Bayes and its variants.
- Use appropriate evaluation metrics.
- Avoid overfitting and underfitting.


