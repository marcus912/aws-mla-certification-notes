# AWS Generative AI & Foundation Models

**Tags:** `#important` `#exam-tip`

## Overview
AWS services for generative AI, foundation models, and large language models (LLMs).

## Amazon Bedrock `#important` `#exam-tip`

**Purpose:** Managed service to access and use foundation models (FMs) via API

**Key Concept:** Pre-trained large models (LLMs, image generation) without managing infrastructure or training models from scratch.

### What Is Bedrock?

**Definition:** Fully managed service providing access to foundation models from multiple AI companies through a single API.

**Foundation Models (FMs):** Large pre-trained models (billions of parameters) that can perform multiple tasks:
- Text generation (chatbots, summarization, Q&A)
- Image generation (text-to-image)
- Embeddings (vector representations for semantic search)

**Key Characteristics:**
- **No infrastructure management** - Serverless, pay-per-use
- **Multiple model providers** - Choose best model for your use case
- **Security & privacy** - Data not used to train models, stays in your AWS account
- **Enterprise-ready** - VPC support, encryption, compliance

### Available Foundation Models `#important`

**Text Models (Large Language Models):**

| Model | Provider | Best For | Context Window |
|-------|----------|----------|----------------|
| **Claude (3.5 Sonnet, 3 Opus, 3 Haiku)** | Anthropic | Long conversations, analysis, coding, safety | 200K tokens |
| **Titan Text (Express, Lite)** | Amazon | Cost-effective text tasks, summarization | 8K-32K tokens |
| **Jurassic-2** | AI21 Labs | Multilingual, long-form generation | 8K tokens |
| **Command** | Cohere | Business writing, RAG applications | 4K tokens |
| **Llama 2/3** | Meta | Open-source, customizable | 4K-8K tokens |

**Image Models:**

| Model | Provider | Best For |
|-------|----------|----------|
| **Stable Diffusion XL** | Stability AI | Text-to-image generation, high quality |
| **Titan Image Generator** | Amazon | Image generation, editing, customization |

**Embedding Models:**
- **Titan Embeddings** - Convert text to vectors for semantic search, RAG
- **Cohere Embed** - Multilingual embeddings

**Exam Tip:** `#exam-tip`
- **Claude** - Most capable, longest context, best for complex tasks
- **Titan** - AWS-owned, cost-effective, good for basic tasks
- **Don't need to memorize all models** - Know categories (text, image, embeddings) and Bedrock provides multiple options

### Key Features

**1. Model Selection** `#exam-tip`
- **Test multiple models** - Compare responses before choosing
- **Model playground** - Test prompts in console
- **Model evaluation** - Built-in benchmarks

**When to choose which model:**
- **Need highest quality, safety:** Claude 3.5 Sonnet or Opus
- **Cost-sensitive, simple tasks:** Titan Text Lite
- **Image generation:** Stable Diffusion XL or Titan Image
- **Embeddings for RAG:** Titan Embeddings

**2. Customization Options**

**Fine-tuning** (Model Customization):
- Train model on your specific data
- Improves performance on domain-specific tasks
- **Requires:** Labeled training data (100s-1000s examples)
- **Use case:** Legal document analysis, medical terminology

**Continued Pre-training:**
- Train model on unlabeled domain data
- Adapts model to specific domain knowledge
- **Use case:** Company-specific knowledge, technical domains

**3. Retrieval Augmented Generation (RAG)** `#exam-tip`

**What is RAG:**
- Retrieve relevant documents from knowledge base
- Provide documents as context to LLM
- LLM generates answer based on retrieved context
- **Benefit:** Up-to-date information without retraining model

**Bedrock Knowledge Bases:**
- Connect to S3, databases, external sources
- Automatic chunking and embedding
- Vector search with OpenSearch Serverless
- **Use case:** Q&A over company documents, customer support

**RAG vs Fine-tuning:** `#important`

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Purpose** | Provide external knowledge | Teach new behavior/style |
| **Data needed** | Documents (unlabeled) | Labeled examples (100s+) |
| **Update frequency** | Real-time (update docs) | Requires retraining |
| **Cost** | Lower (just retrieval) | Higher (training cost) |
| **Use when** | Knowledge changes frequently | Specific task/format needed |
| **Example** | Current product docs, policies | Legal contract generation |

**Exam Scenario:** `#exam-tip`
- **"Answer questions using latest company policies"** → RAG (documents change)
- **"Generate responses in specific legal format"** → Fine-tuning (style/format)

**4. Bedrock Agents** `#exam-tip`

**Purpose:** Build AI agents that can use tools and take actions

**Capabilities:**
- **Break down tasks** into steps
- **Call APIs** to get information or perform actions
- **Use tools** (Lambda functions, APIs)
- **Multi-step reasoning** - Plan and execute complex workflows

**Example:**
- User: "Book me a flight to NYC next week"
- Agent:
  1. Check calendar (Lambda function)
  2. Search flights (API call)
  3. Compare options
  4. Book flight (API call)
  5. Confirm booking

**Use Cases:**
- Customer service automation
- Task automation (data retrieval, report generation)
- Multi-step workflows

**5. Guardrails** `#exam-tip`

**Purpose:** Control model outputs and enforce safety policies

**Features:**
- **Content filters** - Block harmful content (hate speech, violence)
- **Denied topics** - Prevent discussion of specific topics
- **PII filtering** - Redact sensitive information
- **Word filters** - Block profanity, brand names
- **Hallucination detection** - Verify factual accuracy with ground truth

**Use Cases:**
- Customer-facing chatbots (safety)
- Compliance requirements (financial, healthcare)
- Brand protection

**Exam Tip:** Guardrails ensure safe, compliant AI outputs

### Bedrock vs SageMaker Decision Framework `#important` `#exam-tip`

**When to use Amazon Bedrock:**

✅ **Choose Bedrock when:**
- Need to use existing foundation models (LLMs, image generation)
- Want to avoid model training complexity
- Text generation, summarization, chatbots, Q&A
- Image generation from text prompts
- Rapid prototyping and experimentation
- Pre-trained models meet requirements
- RAG for knowledge-based applications
- Don't want to manage ML infrastructure

**Use Cases:**
- Chatbots and virtual assistants
- Document summarization
- Content generation (marketing copy, emails)
- Semantic search
- Code generation
- Customer support automation

**When to use SageMaker:**

✅ **Choose SageMaker when:**
- Need custom ML models from scratch
- Specific algorithms (XGBoost, Random Forest, etc.)
- Tabular data, time series, custom computer vision
- Full control over model architecture
- Proprietary data requires custom training
- Performance optimization critical
- Traditional ML (not just generative AI)

**Use Cases:**
- Fraud detection (tabular data)
- Demand forecasting
- Custom image classification
- Churn prediction
- Recommendation systems (custom algorithms)

**Comparison Table:** `#exam-tip`

| Aspect | Amazon Bedrock | Amazon SageMaker |
|--------|----------------|------------------|
| **Model Type** | Pre-trained foundation models | Custom ML models (any type) |
| **Training** | Optional (fine-tuning) | Required (train from scratch or transfer learning) |
| **Use Cases** | Text generation, chat, image gen | Fraud, forecasting, classification, regression |
| **Data Type** | Primarily text/images | Any (tabular, time series, images, etc.) |
| **Complexity** | Low (API calls) | High (training, tuning, deployment) |
| **Time to Deploy** | Minutes (API ready) | Days/weeks (data prep, training, tuning) |
| **ML Expertise** | Minimal (prompt engineering) | Required (data science, ML engineering) |
| **Infrastructure** | Fully managed, serverless | Managed but configurable (instances, etc.) |
| **Cost Model** | Pay per token/request | Pay for training + inference instances |
| **Customization** | Limited (fine-tuning, prompts) | Complete (any algorithm, architecture) |

**Key Exam Distinction:** `#exam-tip`
- **"Use existing foundation model"** → Bedrock
- **"Train custom model on tabular data"** → SageMaker
- **"Chatbot, summarization, text generation"** → Bedrock
- **"Fraud detection, forecasting, classification"** → SageMaker
- **"Quick deployment without training"** → Bedrock
- **"Need specific algorithm like XGBoost"** → SageMaker

### Use Cases `#exam-tip`

**1. Conversational AI (Chatbots)**
- **Problem:** Build customer service chatbot
- **Solution:** Bedrock with Claude + RAG (company docs)
- **Benefit:** No training, just provide context via RAG

**2. Document Summarization**
- **Problem:** Summarize lengthy reports automatically
- **Solution:** Bedrock with Titan Text or Claude
- **Benefit:** API call, no model training

**3. Content Generation**
- **Problem:** Generate marketing copy, product descriptions
- **Solution:** Bedrock with Claude or Titan Text
- **Benefit:** Consistent, fast content creation

**4. Semantic Search**
- **Problem:** Search company knowledge base by meaning, not keywords
- **Solution:** Bedrock Embeddings + Vector database (OpenSearch)
- **Benefit:** Find relevant docs even with different wording

**5. Code Generation**
- **Problem:** Help developers write code faster
- **Solution:** Bedrock with Claude (good at coding)
- **Benefit:** Generate boilerplate, explain code

### Integration with AWS Services

**Bedrock integrates with:**
- **S3** - Store documents for RAG
- **Lambda** - Serverless application backends
- **OpenSearch Serverless** - Vector search for RAG
- **SageMaker** - Can combine Bedrock FMs with custom SageMaker models
- **CloudWatch** - Monitoring and logging
- **IAM** - Access control
- **VPC** - Private network access

### Pricing `#exam-tip`

**Pricing Models:**

1. **On-Demand (Pay per use):**
   - Charged per **token** (input + output)
   - Token ≈ 0.75 words
   - Example: Claude 3 Haiku ~$0.25 per 1M input tokens
   - **Best for:** Variable usage, prototyping

2. **Provisioned Throughput:**
   - Reserve model capacity (fixed cost)
   - Guaranteed availability and performance
   - **Best for:** Consistent high usage, production workloads

3. **Model Customization:**
   - One-time training cost
   - Storage cost for custom model

**Exam Tip:**
- **On-demand** = pay per token (variable cost)
- **Provisioned** = reserve capacity (fixed cost, predictable)
- No infrastructure costs (serverless)

### Best Practices `#exam-tip`

**1. Prompt Engineering**
- Clear, specific instructions
- Provide examples (few-shot learning)
- Use system prompts for consistent behavior

**2. Use RAG for Knowledge**
- Don't fine-tune for knowledge updates
- Use RAG to provide current information
- Cheaper and more flexible than fine-tuning

**3. Implement Guardrails**
- Always use guardrails for customer-facing applications
- Filter PII, harmful content
- Comply with regulations

**4. Choose Right Model**
- Start with smaller, cheaper models (Titan, Claude Haiku)
- Use larger models (Claude Opus) only when needed
- Test multiple models in playground

**5. Monitor Costs**
- Track token usage
- Optimize prompts (fewer tokens)
- Cache common responses
- Use provisioned throughput for high volume

### Limitations `#exam-tip`

- **Token limits** - Context window varies by model (4K-200K tokens)
- **No fine-grained control** - Can't modify model architecture
- **Model availability** - Limited to available foundation models
- **Not for all ML tasks** - Primarily for generative AI (text, images)
- **Cost at scale** - Token-based pricing can be expensive at very high volumes

### Exam Scenarios `#exam-tip`

| Scenario | Solution | Reasoning |
|----------|----------|-----------|
| "Build chatbot for customer FAQs" | **Bedrock + RAG** | Use FM with company docs, no training needed |
| "Predict customer churn from usage data" | **SageMaker** | Tabular data, custom model (XGBoost) |
| "Generate product descriptions from features" | **Bedrock** | Text generation task, use Titan or Claude |
| "Classify images of manufacturing defects" | **SageMaker or Rekognition Custom Labels** | Computer vision, may need custom model |
| "Summarize customer support tickets" | **Bedrock** | Text summarization, use Claude or Titan |
| "Forecast sales for next quarter" | **SageMaker** | Time series forecasting, use DeepAR |
| "Answer questions using latest company policies" | **Bedrock + RAG** | RAG provides up-to-date context |
| "Detect fraudulent transactions" | **SageMaker** | Tabular data, classification (XGBoost) |
| "Generate images from text descriptions" | **Bedrock** | Use Stable Diffusion XL or Titan Image |

**Key Decision:** `#important`
- **Generative AI (text, images, chat)** → Bedrock
- **Traditional ML (classification, regression, forecasting on structured data)** → SageMaker

## Amazon Q Family `#exam-tip`

**Overview:** Pre-built generative AI assistants for specific use cases.

### Amazon Q Business `#exam-tip`

**Purpose:** Generative AI assistant for enterprise (Q&A over company data)

**Key Features:**
- Chat with enterprise data (documents, wikis, databases)
- 40+ data source connectors (S3, SharePoint, Confluence, Salesforce, Jira, etc.)
- Respects access permissions
- Conversational Q&A
- Document summarization
- Content generation

**Use Cases:**
- Employee self-service (HR, IT policies)
- Knowledge discovery
- Onboarding assistance

**Q Business vs Bedrock:** `#exam-tip`
- **Q Business:** Pre-built assistant, ready to use, enterprise focus
- **Bedrock:** Build your own custom applications with foundation models

### Amazon Q Developer (formerly CodeWhisperer) `#exam-tip`

**Purpose:** AI coding assistant

**Key Features:**
- Code completion (inline suggestions)
- Code generation from comments
- Security vulnerability scanning
- Code explanations
- Supports 15+ languages (Python, Java, JavaScript, etc.)
- IDE integration (VS Code, IntelliJ, etc.)

**Use Cases:**
- Developer productivity
- Learning new APIs/frameworks
- Code review assistance

### Amazon Q Apps `#exam-tip`

**Purpose:** Build generative AI apps from conversations

**Key Features:**
- Create apps from natural language descriptions
- No coding required
- Integrate with Q Business data sources

**Use Cases:**
- Quick business app prototyping
- Department-specific tools

**Exam Tip:** Know Q Business is for enterprise data Q&A, Q Developer is for coding

## Key Comparisons `#exam-tip`

### Bedrock vs Q Business
- **Bedrock:** Foundation model platform (API), build your own apps
- **Q Business:** Pre-built generative AI assistant, ready to use

### Bedrock vs Lex
- **Bedrock:** Conversational AI with LLMs, open-ended conversations, RAG
- **Lex:** Task-oriented bots with intents/slots, structured workflows

### Bedrock vs SageMaker `#important`
- **Bedrock:** Pre-trained foundation models, generative AI (text, images), no training needed
- **SageMaker:** Custom ML models, any algorithm, train from scratch

## Exam Decision Framework `#important`

| Scenario | Solution |
|----------|----------|
| **"Build chatbot"** | **Bedrock (Claude) or Lex** |
| **"Summarize documents"** | **Bedrock** |
| **"Generate content"** | **Bedrock** |
| **"Generate images from text"** | **Bedrock (Stable Diffusion, Titan Image)** |
| **"Semantic search"** | **Bedrock Embeddings + OpenSearch** |
| **"Q&A over company docs"** | **Q Business or Bedrock + RAG** |
| **"Code completion"** | **Q Developer** |
| **"Predict fraud from tabular data"** | **SageMaker (XGBoost)** - NOT Bedrock |
| **"Forecast sales"** | **SageMaker (DeepAR)** - NOT Bedrock |
| **"Image classification"** | **SageMaker or Rekognition** - NOT Bedrock |

**Key Principle:** `#important`
- **Generative AI (text, images, chat)** → Bedrock
- **Traditional ML (classification, regression, forecasting)** → SageMaker
- **Pre-built assistant** → Q family
- **Custom app with foundation models** → Bedrock

## Exam Tips Summary `#exam-tip`

**Bedrock:**
- ✅ Use for generative AI tasks (text generation, chatbots, summarization, image generation)
- ✅ Pre-trained foundation models via API
- ✅ No model training needed (optional fine-tuning)
- ✅ RAG for up-to-date knowledge without retraining
- ✅ Pay per token (on-demand) or provisioned throughput
- ❌ NOT for traditional ML (classification, forecasting on tabular data)

**Key Models to Know:**
- **Claude** - Most capable, longest context (200K tokens)
- **Titan** - AWS-owned, cost-effective
- **Stable Diffusion** - Image generation

**Q Family:**
- **Q Business** - Enterprise data Q&A
- **Q Developer** - Code completion
- **Q Apps** - Build apps from natural language

**Common Exam Mistakes to Avoid:**
- ❌ Using Bedrock for fraud detection (use SageMaker + XGBoost)
- ❌ Using SageMaker for text summarization (use Bedrock)
- ❌ Confusing Q Business with Bedrock (Q is pre-built, Bedrock is platform)

## Related Topics
- [AWS AI Services](./aws-ai-services.md) - Traditional pre-trained AI services
- [Amazon SageMaker](./sagemaker.md) - Custom ML model building
- [Cheat Sheet](./cheat-sheet.md) - Quick reference
