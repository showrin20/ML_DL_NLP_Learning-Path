
### 1. **Classification**
**Definition:** Classification is the task of predicting the class label of given input data.
- Example: In "Where's My Mom?" the butterfly acts as a classifier, using features like size, tusks, tail, etc., to identify the monkey's mom.
- Classifier: A trained model that assigns a category to new data based on features.

**Types of Classifiers:**
- **Discriminative Classifiers:** Predict decision boundaries directly (e.g., Logistic Regression).
- **Generative Classifiers:** Model how data is generated per class (e.g., Naive Bayes).

**Key Concepts:**
- **Training:** Learning from labeled data (training set).
- **Testing:** Evaluating on unseen data (test set).
- **Overfitting:** Model performs well on training but poorly on test data.
- **Underfitting:** Model is too simple and fails on both training and test data.

### 2. **Feature Representation**
Features are properties used by classifiers for prediction.

**Types of Features:**
1. **Bag-of-Words (BoW):**
   - Represents text as word counts.
   - Example: Sentence *"Great movie, great story"* → `f(great) = 2`
   - **Issues:** Sparse matrix, ignores word order.

2. **N-Gram Features:**
   - Considers word sequences.
   - Example: *"not good"* → Bi-gram captures the context better.
   - Typically n ≤ 3 for words, n ≤ 10 for characters.

3. **Lexicon Features:**
   - Use predefined word lists for tasks like sentiment analysis.
   - Example: **LIWC lexicon** labels words like *joyful*, *angry*.

4. **Rule-Based Features:**
   - Custom algorithms to detect patterns.
   - Example: Counting capitalized words in a sentence.

**Feature Refinement:**
- **Binarization:** Treats word presence as binary (0 or 1).
- **Stop Words:** Common words (e.g., *the*, *and*) often removed.
- **Stemming/Lemmatization:** Reduces words to root forms.



### 3. **Probability Review**
**Key Probabilities:**
1. **Prior Probability (P(A)):** Probability of an event without additional information.
   - Example: P(DieRoll=5) = 1/6.

2. **Joint Probability (P(A∩B)):** Probability of two events occurring together.
   - Example: P(A♠) = 1/52.

3. **Conditional Probability (P(A|B)):** Probability of event A given B has occurred.
   - Example: P(A♠|♠) = 1/13.

**Product Rule:** P(A∩B) = P(A|B)P(B).

**Bayes' Theorem:**
P(B|A) = [P(A|B)P(B)] / P(A)

**Purpose:** Allows us to reverse conditional probabilities.

**Conditional Independence:**
- Events A and B are independent given C if: P(A∩B|C) = P(A|C)P(B|C)


# 📊 Performance Metrics in Machine Learning  


1️⃣ **Accuracy**:  
   - Measures the overall correctness of the model.  
   - Formula:  
     \[
     Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
     \]  
   - Best for balanced datasets.  

2️⃣ **Precision** (Positive Predictive Value):  
   - Focuses on how many predicted positives are actually correct.  
   - Formula:  
     \[
     Precision = \frac{TP}{TP + FP}
     \]  
   - Useful when False Positives (FP) need to be minimized (e.g., spam detection).  

3️⃣ **Recall** (Sensitivity/True Positive Rate):  
   - Measures how many actual positives were correctly identified.  
   - Formula:  
     \[
     Recall = \frac{TP}{TP + FN}
     \]  
   - Important when False Negatives (FN) should be minimized (e.g., disease detection).  

4️⃣ **F1-Score**:  
   - Harmonic mean of Precision and Recall.  
   - Formula:  
     \[
     F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
     \]  
   - Best for imbalanced datasets.  

5️⃣ **F-Beta Score**:  
   - Generalized form of F1-Score with a weight factor **β**.  
   - Formula:  
     \[
     F_{\beta} = (1 + \beta^2) \times \frac{Precision \times Recall}{\beta^2 \times Precision + Recall}
     \]  
   - **β > 1**: More emphasis on Recall.  
   - **β < 1**: More emphasis on Precision.  

## 📺 Video Reference  
[Performance Metrics Explained in Hindi](https://youtu.be/5vqk6HnITko?si=nuskNBtieowPpYLm)  


