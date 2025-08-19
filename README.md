# Natural Language Processing Internship - Elevvo Pathways

This project is a part of my **Natural Language Processing Internship** at **Elevvo Pathways**, Egypt.

## Internship Details

- **Company:** Elevvo Pathways, Egypt
- **Internship Period:** July - August 2025

## Sentiment Analysis on Amazon Product Reviews

### Objective
- **Goal**: Build a binary sentiment classifier (positive vs negative) for Amazon product reviews and compare modeling approaches.
- **Scope**: End-to-end pipeline including text cleaning, vectorization, model training (Logistic Regression, Naive Bayes), evaluation, and interpretability/visualization.

### Methodology / Approach
- **Data**: Five Amazon categories (`books`, `ebooks`, `grocery`, `jewelry`, `pc`) combined into a single dataset of review texts with labels.
- **Preprocessing**:
  - Lowercasing, HTML/URL removal, punctuation/numeric stripping
  - Stopword removal with NLTK
  - Lemmatization using NLTK `WordNetLemmatizer`
  - Short-text filter to drop extremely short reviews
- **Vectorization**:
  - TF-IDF and CountVectorizer
  - Unigrams + bigrams, `min_df=2`, `max_df=0.95` (feature space ~1.2k terms)
- **Modeling**:
  - Logistic Regression (`class_weight='balanced'`) for class-imbalance robustness
  - Multinomial Naive Bayes
  - Stratified train/test split with fixed `random_state=42`
- **Evaluation & Analysis**:
  - Metrics: Accuracy, Precision, Recall, F1 (overall) and per-class (Negative/Positive)
  - Confusion matrices; per-class metric comparison plots (focus on Negative class)
  - Feature importance via Logistic Regression coefficients (interpretable model)
  - Word clouds and top word frequency analysis (Positive vs Negative)
  - Category-wise sentiment rates across the five product categories

### Key Results / Observations
- **Best model**: Naive Bayes with CountVectorizer typically achieved the highest F1 on this dataset; Logistic Regression with `class_weight='balanced'` substantially improved Negative-class recall.
- **Class imbalance**: Dataset skews positive (roughly 80/20). Balanced LR and per-class reporting mitigated prior weaknesses on the Negative class.
- **Representative performance**: Models achieved strong performance (F1 in the high‑0.8s in recent runs). Exact numbers may vary slightly by split and preprocessing.
- **Interpretability**:
  - Positive-indicative terms (LR coefficients): e.g., “great”, “love”, “excellent”, “perfect”, “easy”.
  - Negative-indicative terms: e.g., “didnt”, “cheap”, “disappointed”, “weak”, “bad”.
- **Category insights**: Books showed the highest positive rate, while PC/Electronics was comparatively lower, indicating category-dependent sentiment patterns.
- **Qualitative checks**: Sample predictions now correctly classify clearly negative texts; probability outputs reflect higher confidence alignment after preprocessing and class balancing.

### How to Reproduce
1. Open and run `sentiment_analysis_project.ipynb` sequentially (NLTK data is downloaded automatically on first run).
2. Review the performance comparison, per-class tables/plots, and confusion matrices.
3. Inspect interpretability (top LR features) and visualizations (word clouds, frequency bars) for qualitative insights.


