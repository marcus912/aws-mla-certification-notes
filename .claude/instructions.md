# AWS MLA Certification Notes - AI Instructions

## Project Purpose
This project contains study notes for the AWS Machine Learning Associate (MLA) certification exam. All content should be optimized for exam preparation.

## Core Rules for AI Assistants

### 0. Git Workflow `#important`
- **NEVER commit or push changes automatically**
- **NEVER run git add, git commit, or git push commands**
- **DO suggest commit messages** when changes are complete
- **Let the user handle all git operations**
- You may run `git status` or `git diff` to show changes
- User will commit and push when ready

### 0.5. README.md Update Rule `#critical`
**MANDATORY: After ANY content update, you MUST review and update README.md**

When you modify any content file (.md except README.md and TEMPLATE.md):
1. **Count total lines** - Run: `ls *.md | grep -v README.md | grep -v TEMPLATE.md | xargs wc -l`
2. **Count exam tips** - Run: `ls *.md | grep -v README.md | grep -v TEMPLATE.md | xargs grep -o '#exam-tip' | wc -l`
3. **Update README.md stats** - Update "Total Lines" and "Exam Tips" in Repository Stats section
4. **Update structure** - If you added/removed files, update the Structure section
5. **Update coverage** - If you added new algorithms/services, update the coverage lists

**This is NOT optional** - README.md must always reflect the current state of the repository.

**Example workflow:**
- User: "Add notes about X"
- AI: Updates relevant .md file
- AI: **MUST run line count and update README.md**
- AI: Presents all changes together

**Exception:** Only skip README.md update if you're ONLY reading files or answering questions without modifying content.

### 1. Note Style & Format
- **Always use Markdown format** (.md files)
- **Keep notes brief and scannable** - Use bullet points, tables, and short paragraphs
- **No lengthy explanations** - Focus on exam-relevant information only
- **Use concise language** - Remove unnecessary words
- **Active voice preferred** - Direct and clear

### 2. Content Organization
- **Tag every file** with relevant tags: `#core`, `#exam-tip`, `#hands-on`, `#gotcha`, `#important`
- **Cross-reference related topics** - Add links to related notes at the bottom
- **Use consistent structure** - Follow TEMPLATE.md format for new notes
- **Organize by exam domains** - Align with AWS MLA exam structure (Data Prep, Model Dev, Deployment, Monitoring)

### 3. Content Requirements
- **Exam-focused** - Prioritize information likely to appear on the exam
- **Include practical examples** - Real-world scenarios help memorization
- **Highlight common mistakes** - Use `#gotcha` tags for common pitfalls
- **Add exam tips** - Use `#exam-tip` for exam-specific insights
- **Comparison tables** - When multiple options exist, create decision tables
- **"When to use what" sections** - Critical for service selection questions

### 4. Handling Keywords from User
When user provides keywords or topics:

1. **Analyze the keyword** - Determine which existing file(s) it belongs to
2. **Check for existing content** - Search current notes for related information
3. **Update or create** - Either update existing notes or create new file
4. **Reorganize if needed** - Move content to more appropriate files
5. **Update cross-references** - Add links between related topics
6. **Update README.md** - Add new files to the structure
7. **Update cheat-sheet.md** - Add quick reference entries

### 5. Technical Writing Standards
- **Use code blocks** for:
  - Code examples
  - Formulas
  - CLI commands
  - Configuration snippets
- **Use tables** for:
  - Service comparisons
  - Algorithm selection
  - Metric definitions
  - Decision matrices
- **Use lists** for:
  - Features
  - Use cases
  - Best practices
  - Steps in a process

### 6. AWS-Specific Guidelines
- **Service names** - Use official AWS service names (Amazon SageMaker, not just SageMaker)
- **Pricing notes** - Include when relevant for exam (Spot instances, Serverless, etc.)
- **Integration points** - Show how services work together
- **Security** - Include security best practices (IAM, VPC, encryption)
- **Limits and quotas** - Note important service limits

### 7. Content Accuracy
- **Verify technical details** - Ensure accuracy of service capabilities
- **Current information** - AWS services evolve; note if information may be outdated
- **No speculation** - Only include confirmed features and capabilities
- **Cite sources** - Add AWS documentation links when helpful

### 8. Learning Optimization
- **Spaced repetition friendly** - Structure for easy review
- **Progressive disclosure** - Start with overview, then details
- **Memory aids** - Use mnemonics, acronyms, comparisons
- **Visual learning** - Use tables and formatted text for visual scanning
- **Active recall** - Frame information as questions when appropriate

### 9. Multi-language Support
- **English primary** - Default language is English
- **Translations when requested** - Provide Chinese or other languages if user asks
- **Technical terms** - Keep AWS service names in English even in translations
- **Side-by-side format** - Use tables or sections for multi-language content

### 10. File Management
- **Don't create redundant files** - Consolidate related topics
- **Use descriptive filenames** - kebab-case, clear purpose (e.g., `sagemaker-clarify.md`)
- **Keep TEMPLATE.md updated** - Reflect any structural changes
- **Maintain README.md** - **See Rule 0.5** - MUST update after every content change
- **Update cheat-sheet.md** - Add new concepts to quick reference

## Content Priorities (High to Low)

### Must Include (Critical for Exam)
1. Service selection criteria ("When to use X vs Y")
2. Cost optimization strategies
3. Built-in algorithm selection
4. Data format requirements
5. Security best practices
6. Common exam scenarios with solutions
7. Metric definitions and formulas
8. Drift detection concepts
9. Bias and explainability

### Should Include (Important but Secondary)
1. Service features and capabilities
2. Integration patterns
3. Best practices
4. Code examples
5. CLI commands
6. Configuration options

### Nice to Have (If Space Permits)
1. Historical context
2. Alternative approaches
3. Advanced use cases
4. Detailed architecture diagrams
5. Extended examples

## Response Pattern for User Queries

### When user asks: "Add notes about [keyword]"
1. Read existing related files
2. Determine best location for content
3. Create or update appropriate file(s)
4. Add cross-references
5. **MANDATORY: Count lines and exam tips, update README.md (Rule 0.5)**
6. Update cheat-sheet.md if needed
7. Confirm changes to user with file locations and updated stats

### When user asks: "Explain [concept]"
1. Check if concept exists in notes
2. If not, ask where to add it
3. Provide brief, exam-focused explanation
4. Add to appropriate file
5. Use examples and comparisons

### When user asks: "Reorganize notes"
1. Read all existing files
2. Identify overlaps and gaps
3. Propose reorganization plan
4. Execute with user approval
5. Ensure no content is lost
6. Update all cross-references

## Examples of Good vs Bad Content

### ❌ Bad (Too Verbose)
```
Amazon SageMaker is a comprehensive, fully managed machine learning service
that was introduced by AWS to help data scientists and developers prepare,
build, train, and deploy high-quality machine learning models quickly and
efficiently. It provides every developer and data scientist with the ability
to build, train, and deploy machine learning models at scale.
```

### ✅ Good (Brief, Scannable)
```
**Amazon SageMaker:** Fully managed ML service for building, training, and
deploying models at scale.

**Key Features:**
- Managed Jupyter notebooks
- Built-in algorithms (XGBoost, Linear Learner, etc.)
- Automated model tuning
- One-click deployment
```

### ❌ Bad (Missing Context)
```
Use SMOTE for imbalanced datasets.
```

### ✅ Good (Exam-Focused with Context)
```
**SMOTE (Synthetic Minority Over-sampling Technique)** `#exam-tip`
- **Purpose:** Handle class imbalance without losing data
- **How:** Creates synthetic minority samples using interpolation
- **When to use:** Imbalanced classification (fraud, anomaly detection)
- **Important:** Apply only to training data, never test data `#gotcha`
```

## Common Exam Question Patterns to Address

1. **"Which service should you use for..."** → Create decision tables
2. **"How to optimize cost for..."** → List cost-saving strategies
3. **"Model performs poorly on test data..."** → Overfitting/underfitting
4. **"Detect [something] in production..."** → Monitoring solutions
5. **"How to handle [data issue]..."** → Preprocessing techniques
6. **"Real-time vs batch..."** → Deployment pattern selection
7. **"Security best practices for..."** → IAM, VPC, encryption

## Tags Reference

- `#core` - Core exam topic (must know)
- `#exam-tip` - Exam-specific insight or common question
- `#hands-on` - Requires hands-on practice
- `#gotcha` - Common mistake or pitfall
- `#important` - High priority for exam success

## Quality Checklist

Before finalizing any note update, ensure:
- [ ] Brief and scannable (no walls of text)
- [ ] Tagged appropriately
- [ ] Cross-references added
- [ ] Examples included (if relevant)
- [ ] Exam tips highlighted
- [ ] Tables used for comparisons
- [ ] Code blocks for technical content
- [ ] **README.md updated (MANDATORY - see Rule 0.5)** - Line count, exam tips, structure
- [ ] cheat-sheet.md updated (if new concept)
- [ ] No duplicate content across files
- [ ] Markdown formatted correctly

## Continuous Improvement

As the project evolves:
- Consolidate fragmented information
- Remove outdated content
- Enhance cross-referencing
- Add more exam scenarios
- Improve visual structure (tables, formatting)
- Keep aligned with latest AWS features
- Refactor based on user feedback

---

**Remember:** The goal is to help the user pass the AWS MLA exam efficiently. Every piece of content should serve that purpose.
