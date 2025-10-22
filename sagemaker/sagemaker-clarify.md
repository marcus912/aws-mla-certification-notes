# Amazon SageMaker Clarify

**Tags:** `#important` `#exam-tip`

## Overview
Service for detecting bias in ML models and explaining predictions (model explainability).

## Bias Detection `#core`

### What is Bias in SageMaker Clarify Context?

**Definition:** Systematic unfairness in ML model predictions related to sensitive attributes (protected characteristics).

**Sensitive Attributes (Facets):**
- Age, Gender, Race, Ethnicity
- Religion, Disability status
- Marital status, Sexual orientation
- Geographic location

### Types of Bias Clarify Detects

#### Pre-training Bias (Data Bias)
Bias in training data before model is trained.

**Metrics:**
- **Class Imbalance (CI)** - Unequal distribution of labels
  - CI = (n_positive - n_negative) / (n_positive + n_negative)
  - Range: [-1, 1], ideal = 0

- **Difference in Proportions of Labels (DPL)**
  - Compare positive outcome rate between groups
  - DPL = P(y=1|d=1) - P(y=1|d=0)
  - Example: 70% approval rate for group A vs 30% for group B

- **Kullback-Leibler Divergence (KL)**
  - Measures distribution difference between facets

- **Jensen-Shannon Divergence (JS)**
  - Symmetric version of KL divergence

#### Post-training Bias (Model Bias)
Bias in model predictions after training.

**Metrics:**
- **Difference in Predicted Labels (DPL)**
  - Compare prediction rates between groups

- **Disparate Impact (DI)** `#exam-tip`
  - Ratio of positive predictions between groups
  - DI = P(y'=1|d=1) / P(y'=1|d=0)
  - Range: [0, ∞], ideal = 1
  - **Legal threshold:** DI < 0.8 may indicate discrimination (US Equal Employment Opportunity Commission)

- **Difference in Conditional Acceptance (DCA)**
  - Compare acceptance rates given certain conditions

- **Conditional Demographic Disparity (CDD)**
  - Measures prediction disparity within subgroups

### Bias Example `#exam-tip`

**Scenario:** Loan approval model
- **Sensitive attribute:** Gender (Male/Female)
- **Pre-training bias detected:**
  - Training data: 80% male applicants approved, 50% female applicants approved
  - DPL = 0.80 - 0.50 = 0.30 (indicates bias in historical data)
- **Post-training bias detected:**
  - Model predictions: 75% male approved, 45% female approved
  - Disparate Impact = 0.45/0.75 = 0.60 (below 0.8 threshold → potential discrimination)
- **Action:** Retrain with balanced data, apply fairness constraints, or adjust decision threshold

## Model Explainability

### SHAP (SHapley Additive exPlanations) `#exam-tip`
- **Purpose:** Explain individual predictions
- **How it works:**
  - Calculates contribution of each feature to prediction
  - Based on game theory (Shapley values)
  - Shows positive/negative impact of features
- **Output:**
  - Feature importance scores
  - Visualization showing which features pushed prediction up/down
- **Example:** Why was loan denied? SHAP shows: low_income (-0.3), high_debt_ratio (-0.2), young_age (-0.1)

### Partial Dependence Plots (PDP)
- Shows relationship between feature and predictions
- Marginal effect of features

## Integration Points `#hands-on`

### When to Use Clarify
1. **During model development** - Analyze training data for bias
2. **Before deployment** - Validate model fairness
3. **In production** - Continuous monitoring with Model Monitor
4. **For compliance** - Generate explainability reports

### SageMaker Integration
- **Clarify Processing Jobs** - Run bias analysis
- **Model Monitor integration** - Continuous bias monitoring
- **SageMaker Studio** - Visual bias reports
- **Clarify Explainability** - Generate SHAP values for predictions

## Exam Tips `#exam-tip`
- **Pre-training bias:** Data bias (before training)
- **Post-training bias:** Model prediction bias (after training)
- **Disparate Impact < 0.8:** Legal red flag
- **SHAP:** For explaining individual predictions
- **Use Clarify for:** Regulated industries (finance, healthcare, hiring)
- **Bias metrics:** Know difference between DPL, DI, CI

## Gotchas `#gotcha`
- Clarify requires explicit facet (sensitive attribute) specification
- Bias detection is statistical - doesn't guarantee legal compliance
- Must define "favorable outcome" for bias metrics
- SHAP computation can be expensive for large models

## Related Topics
- [Model Training & Evaluation](../core-ml/model-training-evaluation.md)
- [MLOps & Deployment](../mlops/mlops-deployment.md)
- [Amazon SageMaker](./sagemaker.md)
