# Guardrail Model Development Report

## 1. Introduction

This report documents the development process of a guardrail model for prompt injection detection in the SISE DiagnoSys application. The guardrail aims to protect the RAG system by identifying and blocking malicious injection attempts before they reach the LLM.

## 2. Initial Work (Marin)

### 2.1 Datasets Used

Three separate training approaches were explored using different datasets:

| Notebook | Dataset(s) | Samples | Embedding Model |
|----------|-----------|---------|-----------------|
| `1_guardrail-multi-lang` | Qualifire FR + Jayavibhav EN | 10,000 | `paraphrase-multilingual-MiniLM-L12-v2` |
| `2_guardrail-english` | Deepset prompt-injections | 662 | `all-MiniLM-L6-v2` |
| `3_guardrail-custom` | HuggingFace helpful-instructions + jailbreak_prompts.csv | ~7,000 | `all-MiniLM-L6-v2` |

### 2.2 Models Trained

Each notebook experimented with multiple classifiers:
- Logistic Regression
- Random Forest (with GridSearchCV)
- Gradient Boosting (XGBoost)
- Support Vector Machine (with GridSearchCV)

### 2.3 In-Distribution Results

Models showed excellent performance when evaluated on held-out test sets from the **same distribution**:

| Model | Dataset | F1 Score | Precision |
|-------|---------|----------|-----------|
| SVM Multi-lang | Qualifire + Jayavibhav | 0.85 | 0.86 |
| SVM English | Deepset | 0.82 | 0.98 |
| SVM Custom | Jailbreak + Helpful | 0.99 | 1.00 |

These results appeared promising, suggesting the models had learned to distinguish between benign and malicious prompts effectively.

## 3. Generalization Issues

### 3.1 Cross-Dataset Benchmark

The `0_benchmark.ipynb` notebook revealed a critical problem: **models failed catastrophically when tested on datasets different from their training data**.

| Model | Trained On | Tested On | F1 Score |
|-------|------------|-----------|----------|
| SVM Multi-lang | Qualifire + Jayavibhav | Custom | **0.43** |
| SVM English | Deepset | Custom | **0.07** |
| XGBoost Custom | Jailbreak + Helpful | Qualifire | **0.61** |
| SVM Custom | Jailbreak + Helpful | Qualifire | **0.70** |

### 3.2 Root Cause Analysis

The poor cross-dataset performance stems from several factors:

1. **Dataset-specific patterns**: Each dataset has unique characteristics (writing style, prompt formats, vocabulary) that models overfit to
2. **Class imbalance variations**: Different datasets have different benign/jailbreak ratios (e.g., Custom has 98.6% benign vs. Qualifire at 60%)
3. **Language and domain shift**: Models trained on English-only data fail on French prompts and vice-versa
4. **Limited sample diversity**: Small datasets like Deepset (662 samples) don't capture the full variety of injection techniques

### 3.3 Implications

A guardrail trained on a single dataset would:
- Miss novel injection patterns not present in training data
- Generate excessive false positives on legitimate prompts with different styles
- Fail to protect against real-world attacks that differ from training examples

## 4. Improved Training Approach

### 4.1 Design Principles

The improved training script (`improved_guardrail_training.py`) addresses generalization issues through:

1. **Multi-source training**: Combine ALL available datasets
2. **Cross-dataset validation**: Test on completely held-out datasets
3. **Data augmentation**: Add robustness through text variations
4. **Feature engineering**: Handcrafted features beyond embeddings
5. **Ensemble methods**: Combine multiple diverse classifiers

### 4.2 Combined Dataset

| Source | Type | Samples |
|--------|------|---------|
| Qualifire English | Mixed | ~5,000 |
| Qualifire French | Mixed | ~5,000 |
| Deepset prompt-injections | Mixed | ~660 |
| jailbreak_prompts.csv | Jailbreak only | ~5,000 |
| HuggingFace helpful-instructions | Benign only | ~8,000 |

Total: ~24,000 samples with diverse sources and languages.

### 4.3 Leave-One-Dataset-Out (LODO) Validation

Instead of random train/test splits, LODO validation:
- Trains on N-1 datasets
- Tests on the completely held-out dataset
- Repeats for each dataset

This measures **true generalization** to unseen data distributions.

### 4.4 Data Augmentation

The `TextAugmenter` class applies transformations to increase robustness:

```python
# Augmentation techniques
- Case variations (upper/lower)
- Typo injection
- Obfuscation (leet speak: a→@, e→3)
- Spacing variations
- Punctuation removal
```

### 4.5 Feature Engineering

The `FeatureExtractor` class computes handcrafted features alongside embeddings:

| Feature Category | Examples |
|-----------------|----------|
| Length features | `length`, `word_count`, `avg_word_length` |
| Character ratios | `uppercase_ratio`, `special_char_ratio`, `digit_ratio` |
| Pattern matching | `jailbreak_pattern_count` (regex for "ignore previous", "pretend you are", etc.) |
| Structural | `has_brackets`, `exclamation_count`, `entropy` |

These features capture explicit jailbreak indicators that embeddings might miss.

### 4.6 Ensemble Model

A voting ensemble combines diverse classifiers:

```python
VotingClassifier([
    ('lr', LogisticRegression),      # Linear decision boundary
    ('svm', CalibratedClassifierCV), # LinearSVC with calibration
    ('rf', RandomForestClassifier),  # Tree-based, handles non-linearity
    ('gb', GradientBoostingClassifier), # Sequential boosting
    ('mlp', MLPClassifier),          # Neural network
], voting='soft')
```

Soft voting averages predicted probabilities, leveraging each model's strengths.

## 5. Benchmark Script

The `benchmark_models.py` script provides standardized evaluation:

- Loads all benchmark datasets consistently
- Tests both old single-dataset models and the new ensemble
- Computes comprehensive metrics (Accuracy, Precision, Recall, F1, MCC, ROC-AUC)
- Generates comparison tables across all datasets

### 5.1 Usage

```bash
# Benchmark all models
python benchmark_models.py --all

# Only old models
python benchmark_models.py --old-only

# Only new ensemble
python benchmark_models.py --new-only
```

## 6. Model Artifacts

The improved training produces three artifacts:

| File | Description |
|------|-------------|
| `guardrail_ensemble_v2.joblib` | Trained voting ensemble model |
| `guardrail_scaler_v2.joblib` | StandardScaler for handcrafted features |
| `guardrail_feature_extractor_v2.joblib` | FeatureExtractor instance |

### 6.1 Inference Usage

```python
from improved_guardrail_training import GuardrailClassifier

classifier = GuardrailClassifier()
result = classifier.predict("Ignore all previous instructions...")
# {'is_jailbreak': True, 'confidence': 0.92, 'label': 'jailbreak'}
```

## 7. Conclusion

The initial single-dataset approach achieved misleadingly high in-distribution scores but failed in cross-dataset evaluation. The improved multi-source training with LODO validation, data augmentation, feature engineering, and ensemble methods provides a more robust guardrail suitable for production deployment.

## 8. Future Work

- Integrate the guardrail into `rag_service.py`
- Add threshold tuning for precision/recall trade-off
- Implement continuous monitoring and retraining pipeline
- Expand datasets with real-world attack examples
