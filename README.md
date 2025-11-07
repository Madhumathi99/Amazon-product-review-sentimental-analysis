Aspect-Based Sentiment Analysis with Sarcasm Detection on Amazon Product Reviews Using Classical and Transformer-Based Models
Aim and Motivation
Improve sentiment analysis by combining aspect-based sentiment analysis (ABSA) with sarcasm detection.
Traditional models fail to capture context nuances, especially sarcasm, which can invert the intended sentiment.
Focus on Amazon product reviews, where users often express implicit or sarcastic opinions.
Research Objectives
Compare performance of traditional models vs. transformer models (DistilBERT) in handling sarcasm in sentiment classification.
Identify the impact of sarcasm on sentiment misclassification, especially for ambiguous or negative reviews.
Evaluate the effect of data preprocessing and class balancing techniques.
Build a robust classification pipeline capturing both aspects and subtle tones.
Research Questions
How does sarcasm influence sentiment detection accuracy?
Can transformer models like DistilBERT outperform classical models?
What is the effect of text cleaning and balanced datasets on performance?
Dataset and Preprocessing
Used 50,000 Amazon reviews from 5 categories:
Electronics, Books, Home & Kitchen, Cell Phones & Accessories, Sports & Outdoors.
Preprocessing steps:
Emoji normalization, text cleaning (punctuation, stopword, number removal), and review labeling.
Sentiment mapping:  1–2: Negative, 3: Neutral, 4–5: Positive.
Created 3 versions of the dataset:
df_balanced_3class: Equal distribution (Negative, Neutral, Positive)
df_binary: Neutral merged with Positive (binary)
df_full_weighted: Original distribution weighted
Aspect-Based Sentiment Analysis (ABSA):
Used pretrained model yangheng/deberta-v3-base-absa-v1.1.
Extracted product aspects (e.g., battery, price) and tagged their associated polarity.
Helped identify feature-based sentiment often hidden in longer, mixed-topic reviews.
Sarcasm Detection
Applied helinivan/english-sarcasm-detector transformer model.
Sarcasm detected at both review and aspect levels.
Findings
Sarcasm appeared in ~3.8% overall reviews.
Most in negative reviews (8.42%), least in positive (3.21%).
Categories with most sarcasm: Cell Phones & Electronics.
Modeling Approaches
A. Classical ML Models (TF-IDF features):
Models used: Logistic Regression, SVM, Naive Bayes, Random Forest.
Trained on df_balanced_3class.
Results: ~66–68% accuracy, struggled particularly with Neutral class.
B. Transformer Models (DistilBERT):
Fine-tuned across all 3 datasets.
Key configurations: 3 epochs, CrossEntropyLoss (weighted where needed).
Performance Summary:
Model Version	Accuracy	F1-Score
DistilBERT_Binary	96.6%	0.966
DistilBERT_Balanced	71.3%	0.716
DistilBERT_Weighted	89.1%	0.894
Binary performed best due to reduced complexity
Weighted overcame imbalance
Neutral sentiment remained hardest to detect
Impact of Sarcasm on ML Models
Sarcasm, especially when positive words were used to express negative sentiment, led to many misclassifications.
SVM handled sarcasm best among classical models.
Sarcasm detection improved false negative rate in sentiment predictions.
Sarcasm Detector Performance:
Model	            Accuracy
SVM	                96.1%
Logistic Regression	94.1%
Naive Bayes	        88.9%
Key Results and Observations
Transformer models outperforms classical ML in all tasks.
Sarcasm detection significantly improves sentiment accuracy, especially for negative reviews.
Aspect-sentiment mapping helps uncover deeper insights about customer opinions, beyond surface text.
Binary classification (Positive vs Negative) is easier and more reliable than multi-class.
Conclusion
Combining ABSA + sarcasm detection + DistilBERT provides a robust, real-world sentiment analysis system.
Helps businesses get accurate customer sentiment insights even when language is nuanced or sarcastic.
Ideal for customer feedback systems, brand monitoring, and review analytics.
Future Recommendations
Use explainability tools (e.g., SHAP) for interpreting model outputs.
Adapt sarcasm detection into real-time sentiment engines.
Extend approach to different domains beyond e-commerce.
