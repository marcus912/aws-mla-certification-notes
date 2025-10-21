# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **study notes repository** for the AWS Certified Machine Learning Associate (MLA-C01) certification exam. It contains 13+ markdown files with exam-focused content covering all 4 AWS MLA exam domains.

**Key characteristics:**
- Pure documentation repository (no executable code, no build/test commands)
- All content in markdown (.md) files
- Optimized for exam preparation (brief, scannable, exam-focused)
- Uses tags: `#core`, `#exam-tip`, `#hands-on`, `#gotcha`, `#important`

## Content Structure

### Core Organization (3 categories)

**1. Core ML Concepts** (ml-fundamentals.md, model-training-evaluation.md, feature-engineering.md)
- Foundational ML theory and practice
- Independent of AWS services

**2. AWS ML/AI Services** (8 files covering SageMaker, AI services, data services, MLOps, security)
- AWS-specific implementation details
- Service selection criteria crucial for exam

**3. Quick References** (study-guide.md, cheat-sheet.md)
- study-guide.md: 10-week study plan, domain breakdown, exam strategy
- cheat-sheet.md: Quick decision tables ("When to use what")

### File Relationships

- **sagemaker.md** → Central hub for custom ML (links to algorithms, deployment, monitoring)
- **aws-ml-algorithms.md** → Details on 17 built-in SageMaker algorithms
- **mlops-deployment.md** → Deployment patterns (Endpoints, Pipelines, Monitoring, K8s/Kubeflow)
- **aws-ai-services.md** → Pre-trained traditional AI (Comprehend, Rekognition, Lex, etc.)
- **aws-generative-ai.md** → Foundation models (Bedrock: Claude, Titan, Stable Diffusion; Amazon Q; Agents)
- **sagemaker-clarify.md** → Bias detection and explainability
- **data-services.md** → Data pipeline services (S3, Glue, Athena, EMR, Kinesis, Lake Formation)
- **security.md** → IAM, VPC, encryption, compliance

## Critical Workflow Rules

### MANDATORY: README.md Update After Content Changes

**After ANY modification to content files (.md except README.md and TEMPLATE.md):**

1. **Count total lines:**
   ```bash
   ls *.md | grep -v README.md | grep -v TEMPLATE.md | xargs wc -l
   ```

2. **Count exam tips:**
   ```bash
   ls *.md | grep -v README.md | grep -v TEMPLATE.md | xargs grep -o '#exam-tip' | wc -l
   ```

3. **Update README.md "Repository Stats" section:**
   - Total Lines
   - Exam Tips count
   - If adding new algorithms/services: update coverage lists
   - If adding/removing files: update Structure section

**This is NOT optional.** README.md must reflect current state.

### Git Workflow

- **NEVER commit automatically**
- **DO suggest commit messages** when changes complete
- User handles all git operations (add, commit, push)
- You may run `git status` or `git diff` to show changes

### Content Reorganization Authority

You have permission to reorganize proactively when it improves structure:

**Reorganize immediately when:**
- File exceeds 500 lines → Split into logical sub-topics
- Content doesn't fit existing files → Create new file
- Same content duplicated across 3+ files → Consolidate
- User adds topic requiring new file

**After reorganization:**
1. Update all cross-reference links (use Grep to find `./filename.md` references)
2. Update "Related Topics" sections in affected files
3. Update README.md Structure section
4. Update README.md stats (line count, file count)
5. Verify no broken links

## Writing Style Guidelines

### Format Rules

**Brief and scannable:**
- Use bullet points over paragraphs
- Short sentences (< 20 words)
- Tables for comparisons
- Code blocks for technical content

**Structure pattern:**
```markdown
## Service/Concept Name `#exam-tip`
**Purpose:** One-line description

**Key Features:**
- Feature 1
- Feature 2

**When to Use:**
- Scenario 1
- Scenario 2

**Exam Scenarios:** `#exam-tip`
- "Question pattern?" → Answer
```

### Content Priorities

**Must include (exam-critical):**
- Service selection criteria ("When to use X vs Y")
- Decision tables with comparison
- Cost optimization strategies
- Common exam scenarios with solutions
- Security best practices

**Should include:**
- Service features and capabilities
- Integration patterns
- Code examples with context

**Avoid:**
- Lengthy explanations
- Marketing language
- Speculation about future features
- Generic development advice

### Tagging System

- `#core` - Core exam topic (must know)
- `#exam-tip` - Exam-specific insight or common question pattern
- `#hands-on` - Requires hands-on lab practice
- `#gotcha` - Common mistake or pitfall
- `#important` - High priority for exam success

## Common Tasks

### Adding New Content

**User request: "Add notes about [X]"**

1. Read related files to understand existing structure
2. Determine best location (or if new file needed)
3. Check if reorganization needed (file >500 lines? Content scattered?)
4. Add content following TEMPLATE.md structure
5. Add cross-references to related topics
6. **MANDATORY: Update README.md stats**
7. Update cheat-sheet.md if new decision pattern
8. Suggest commit message

### Answering Questions

**User request: "Explain [concept]"**

1. Check if concept exists in notes
2. Provide brief, exam-focused explanation
3. Ask if they want it added to notes
4. If yes, follow "Adding New Content" workflow

### Updating Existing Content

1. Read current content
2. Make updates maintaining consistent style
3. Check for affected cross-references
4. **MANDATORY: Update README.md stats if line count changed**
5. Suggest commit message

## Repository-Specific Patterns

### Cross-Referencing Format
- Relative links: `[SageMaker](./sagemaker.md)`
- Section links: `[Model Monitor](./mlops-deployment.md#model-monitoring)`
- Always update "Related Topics" section at bottom of files

### Exam Scenario Format
```markdown
**Exam Scenarios:** `#exam-tip`
- **"Question from exam perspective?"** → Concise answer
```

### Comparison Table Format
```markdown
| Scenario | Solution | Why |
|----------|----------|-----|
| Pattern 1 | Service A | Reason |
| Pattern 2 | Service B | Reason |
```

### Code Example Format
```python
# Brief description of what this does
code_example()  # Comment on key line
```

## Quality Checklist Before Completing Work

- [ ] Content is brief and scannable
- [ ] Tables used for comparisons
- [ ] Code blocks for technical content
- [ ] Appropriate tags applied
- [ ] Cross-references added to "Related Topics"
- [ ] Exam tips highlighted with `#exam-tip`
- [ ] **README.md stats updated (MANDATORY)**
- [ ] cheat-sheet.md updated if new concept/pattern
- [ ] No duplicate content across files
- [ ] File sizes reasonable (<500 lines)
- [ ] Commit message suggested

## AWS Certification Exam Context

**Exam domains (align content with these):**
1. Data Engineering (20%) - Data prep, feature engineering, pipelines
2. Exploratory Data Analysis (24%) - Analysis, visualization, statistics
3. Modeling (36%) - Algorithm selection, training, tuning, evaluation
4. ML Implementation & Operations (20%) - Deployment, monitoring, security

**Common exam question patterns:**
- "Which service for..." → Service selection
- "Optimize cost..." → Cost strategies
- "Real-time vs batch..." → Deployment pattern
- "Model performs poorly..." → Debugging/tuning
- "Detect drift..." → Monitoring solutions
- "Security requirement..." → IAM/VPC/encryption

These patterns should appear throughout notes as "Exam Scenarios" sections.
