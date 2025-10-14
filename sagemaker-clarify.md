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

## SageMaker Clarify 偏差检测 (中文解释)

### 什么是偏差 (Bias)?

**定义:** 机器学习模型预测中与敏感属性（受保护特征）相关的系统性不公平。

**敏感属性例子:**
- 年龄、性别、种族、民族
- 宗教、残疾状况
- 婚姻状况、性取向
- 地理位置

### SageMaker Clarify 检测的偏差类型

#### 训练前偏差 (Pre-training Bias)
训练数据中存在的偏差（模型训练之前）

**指标:**
- **类别不平衡 (Class Imbalance, CI)** - 标签分布不均
- **标签比例差异 (DPL)** - 比较不同组之间的正向结果率
  - 例子：A组批准率70% vs B组批准率30%

#### 训练后偏差 (Post-training Bias)
模型预测中的偏差（训练之后）

**指标:**
- **预测标签差异 (DPL)** - 比较组间预测率
- **不同影响 (Disparate Impact, DI)** - 组间正向预测的比率
  - 理想值 = 1
  - DI < 0.8 可能表明存在歧视（美国法律阈值）

### 实际例子

**场景:** 贷款审批模型
- **敏感属性:** 性别（男/女）
- **检测到训练前偏差:**
  - 训练数据：80%男性申请人获批，50%女性申请人获批
  - 标签比例差异 = 0.30（历史数据存在偏差）
- **检测到训练后偏差:**
  - 模型预测：75%男性获批，45%女性获批
  - 不同影响 = 0.45/0.75 = 0.60（低于0.8阈值 → 可能存在歧视）
- **行动:** 使用平衡数据重新训练、应用公平性约束、或调整决策阈值

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
- [Model Training & Evaluation](./model-training-evaluation.md)
- [MLOps & Deployment](./mlops-deployment.md)
- [Amazon SageMaker](./sagemaker.md)
