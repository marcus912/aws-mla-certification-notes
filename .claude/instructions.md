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

When you modify any content file (.md except README.md, CLAUDE.md, and guides/TEMPLATE.md):
1. **Count total lines** - Run: `find . -name "*.md" -not -path "./README.md" -not -path "./CLAUDE.md" -not -path "./guides/TEMPLATE.md" | xargs wc -l`
2. **Count exam tips** - Run: `find . -name "*.md" -not -path "./README.md" -not -path "./CLAUDE.md" -not -path "./guides/TEMPLATE.md" | xargs grep -o '#exam-tip' | wc -l`
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
- **Organize in folders** - Place files in appropriate folders: core-ml/, sagemaker/, aws-services/, mlops/, security/, guides/
- **Keep guides/TEMPLATE.md updated** - Reflect any structural changes
- **Maintain README.md** - **See Rule 0.5** - MUST update after every content change
- **Update guides/cheat-sheet.md** - Add new concepts to quick reference

### 11. Autonomous Content Reorganization `#important`

**You HAVE PERMISSION to reorganize notes proactively** when it improves clarity and structure.

**Current Folder Structure:**
- `core-ml/` - ML fundamentals (3 files)
- `sagemaker/` - SageMaker services (5 files)
- `aws-services/` - AWS AI/ML services (4 files)
- `mlops/` - MLOps workflows (4 files)
- `security/` - Security topics (3 files)
- `guides/` - Study resources (3 files: study-guide.md, cheat-sheet.md, TEMPLATE.md)

#### When to Reorganize (Do it proactively)

**Create new note files when:**
- A topic within a note grows beyond 200 lines and deserves its own file
- New AWS service/concept needs dedicated coverage
- Multiple files reference the same complex concept repeatedly
- Content becomes too diverse for current file (e.g., mixing algorithms with deployment)
- **Place new files in the appropriate folder** based on topic

**Rename note files when:**
- Current name no longer reflects the content (e.g., file grew to cover more topics)
- Better name improves discoverability
- Aligns better with AWS official terminology

**Move content between notes when:**
- Content is in wrong file based on topic/domain
- Better logical grouping improves learning flow
- Reduces duplication across files
- New terminology makes old organization obsolete

**Split existing notes when:**
- File exceeds 500 lines
- Contains 3+ distinct major topics
- Difficult to navigate or find specific information
- New topics make current file scope too broad

#### Reorganization Process

**For MINOR changes (no new files):**
1. **Just do it** - Move content, update cross-references
2. Update README.md (Rule 0.5)
3. Inform user of changes: "Reorganized X content from A to B for better clarity"

**For MAJOR changes (new files, renames, splits):**
1. **Announce intent first** - "I'm creating a new file X because..." or "I'm splitting A into A and B because..."
2. Execute the reorganization
3. **Ensure no content loss** - Verify all content moved successfully
4. **Update all cross-references** - Fix links in all affected files
5. **Update README.md** - Add new files to Structure section
6. **Update cheat-sheet.md** - Adjust references if needed
7. **Summarize changes** - List all files created/renamed/modified

#### Required Actions After Reorganization

**Always do:**
- [ ] Search and replace all cross-reference links (use Grep tool)
- [ ] Update "Related Topics" sections in all affected files (use ../folder/ for cross-folder links)
- [ ] Update README.md Structure section
- [ ] Update README.md stats (line count, file count)
- [ ] Verify no broken links remain
- [ ] Ensure files are in correct folders
- [ ] Ensure consistent tagging across reorganized content

**Never do:**
- ❌ Delete content without moving it elsewhere
- ❌ Rename files without updating cross-references
- ❌ Create files that duplicate existing content
- ❌ Reorganize without updating README.md

#### Examples of Good Reorganization

**Example 1: Creating new file**
```
Situation: sagemaker/sagemaker.md has 600 lines, including 150 lines on SageMaker Pipelines
Action: Create mlops/mlops-pipelines.md, move Pipelines content there
Rationale: MLOps deserves dedicated file in mlops/ folder
Update: Fix cross-references (../mlops/mlops-pipelines.md), update README.md structure
```

**Example 2: Moving content**
```
Situation: Regularization concepts scattered across 3 files
Action: Consolidate all regularization content into core-ml/model-training-evaluation.md
Rationale: Single source of truth for regularization concepts
Update: Remove from other files, add cross-references using ../core-ml/ paths
```

**Example 3: Splitting file**
```
Situation: aws-services/aws-ai-services.md is 800 lines covering 16 services
Action: Split into:
  - aws-services/aws-ai-services-nlp.md (NLP services)
  - aws-services/aws-ai-services-vision.md (Vision services)
  - aws-services/aws-ai-services-other.md (Remaining services)
Rationale: Easier navigation, grouped by use case, all in aws-services/ folder
Update: Create 3 files, update all cross-references (same folder = ./), update README.md
```

**Example 4: Moving to correct folder**
```
Situation: New file deployment.md created in root
Action: Move to mlops/mlops-deployment.md (correct folder)
Rationale: MLOps content belongs in mlops/ folder
Update: Search all files for "deployment.md" links, replace with "../mlops/mlops-deployment.md"
```

#### Decision Framework

**Ask yourself:**
- Will this make finding information easier?
- Does this reduce duplication?
- Is the new structure more logical for exam preparation?
- Are file sizes becoming unwieldy (>500 lines)?
- Would AWS documentation structure it this way?

If YES to 2+ questions → Reorganize proactively

If unsure → Propose to user first: "I suggest reorganizing X because Y. Proceed?"

#### Triggers for Reorganization

**Automatic triggers (reorganize immediately):**
- User adds new topic that doesn't fit existing files cleanly
- File exceeds 500 lines
- Same content appears in 3+ files
- User explicitly requests reorganization

**Consider reorganization when:**
- Adding 100+ lines of new content
- Creating new major section in existing file
- User asks "where should I find X?" (discoverability issue)
- Cross-references become complex web

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
2. Determine best location for content **and appropriate folder**
3. **Check if reorganization needed (Rule 11):**
   - Would new content exceed 500 lines in target file? → Consider split
   - Does keyword deserve its own file? → Create new file in appropriate folder
   - Is content scattered across files? → Consolidate first
4. Create or update appropriate file(s) in correct folder
5. Add cross-references (use ../folder/ for cross-folder links)
6. **MANDATORY: Count lines and exam tips, update README.md (Rule 0.5)**
7. Update guides/cheat-sheet.md if needed
8. Confirm changes to user with file locations and updated stats

### When user asks: "Explain [concept]"
1. Check if concept exists in notes
2. If not, ask where to add it
3. Provide brief, exam-focused explanation
4. Add to appropriate file
5. Use examples and comparisons

### When user asks: "Reorganize notes"
**Follow Rule 11 (Autonomous Content Reorganization)**
1. Read all existing files to assess current structure
2. Identify overlaps, gaps, and improvement opportunities
3. For MAJOR changes: Announce reorganization plan first
4. For MINOR changes: Execute immediately
5. Execute reorganization (create/rename/move/split as needed)
6. Update all cross-references (use Grep to find all links)
7. Update README.md structure and stats (Rule 0.5)
8. Verify no content lost, no broken links
9. Summarize all changes to user

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
- [ ] **File organization optimal (Rule 11)** - No files >500 lines, content in logical places

## Continuous Improvement

As the project evolves:
- **Proactively reorganize** per Rule 11 when structure can be improved
- Consolidate fragmented information
- Remove outdated content
- Enhance cross-referencing
- Add more exam scenarios
- Improve visual structure (tables, formatting)
- Keep aligned with latest AWS features
- Refactor based on user feedback
- Monitor file sizes (split when >500 lines)

---

**Remember:** The goal is to help the user pass the AWS MLA exam efficiently. Every piece of content should serve that purpose.
