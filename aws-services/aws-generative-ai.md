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

**4. Bedrock Agents** `#important` `#exam-tip`

**Purpose:** Build AI agents that can use tools, take actions, and orchestrate complex multi-step workflows

**Key Concept:** Agents combine foundation models with the ability to call APIs, access data, and execute actions autonomously.

#### What are Bedrock Agents?

**Definition:** Fully managed capability that allows LLMs to execute multi-step tasks by:
- Breaking down user requests into steps
- Calling APIs and Lambda functions
- Accessing Knowledge Bases for information
- Reasoning about next actions
- Orchestrating complex workflows

**Key Characteristics:**
- **Autonomous reasoning** - Agent decides which tools to use and when
- **Multi-step execution** - Chains multiple actions together
- **Tool integration** - Lambda functions, APIs, Knowledge Bases
- **Natural language interface** - Users interact in plain language
- **Managed orchestration** - AWS handles the workflow logic

#### Agent Components `#exam-tip`

**1. Foundation Model**
- Base LLM that powers the agent (Claude, Titan, etc.)
- Provides reasoning and language understanding
- Determines which actions to take

**2. Instructions**
- Natural language description of what the agent does
- Agent's role, capabilities, and behavior
- Example: "You are a travel assistant that helps users book flights and hotels"

**3. Action Groups** `#important`
- **Definition:** Collections of APIs/functions the agent can call
- **Implementation:** Lambda functions or API schemas
- **Purpose:** Define available tools/actions

**Action Group Components:**
- **API Schema** - OpenAPI spec defining available functions
- **Lambda Function** - Executes the action
- **Description** - Helps agent decide when to use this action

**Example Action Group:**
```json
{
  "actionGroupName": "FlightBookingActions",
  "description": "Actions for searching and booking flights",
  "actionGroupExecutor": {
    "lambda": "arn:aws:lambda:us-east-1:123456789012:function:flight-booking"
  },
  "apiSchema": {
    "payload": "OpenAPI 3.0 schema with functions: searchFlights, bookFlight, cancelFlight"
  }
}
```

**4. Knowledge Bases** (Optional)
- Connect agent to Bedrock Knowledge Bases
- Agent can retrieve information from documents
- Combines RAG with agentic workflows
- **Use case:** Agent answers questions using company docs, then takes actions

**5. Guardrails** (Optional)
- Apply content filters and safety policies
- Ensure safe, compliant agent behavior

#### Agent Workflow (Orchestration) `#exam-tip`

**How Agents Work:**
1. **User Request** - Natural language input
2. **Agent Planning** - LLM analyzes request, breaks into steps
3. **Action Selection** - Agent decides which Action Group/function to call
4. **Execution** - Invokes Lambda functions, queries Knowledge Bases
5. **Iteration** - Uses results to determine next action, repeats until complete
6. **Response** - Synthesizes results into natural language

**Example:** User: "Book flight to NYC next week" → Agent: checks calendar → searches flights → books → responds

#### Agent Versions and Aliases `#important` `#exam-tip`

**Agent Versioning:**

**Purpose:** Manage different versions of your agent for safe deployment and testing.

**Key Concepts:**
- **Working Draft** - Editable version, always exists
- **Versions** - Immutable snapshots (v1, v2, v3, etc.)
- **Aliases** - Pointers to specific versions

**Version Lifecycle:**

1. **Working Draft**
   - Active development version
   - Can be edited and tested
   - Not suitable for production
   - Always available

2. **Create Version**
   - Snapshot the Working Draft
   - Creates immutable version (e.g., v1)
   - Cannot be modified
   - Can be used in production

3. **Aliases**
   - Named pointers to versions
   - Can be updated to point to different versions
   - Enable safe deployment patterns

**Agent Aliases** `#exam-tip`

**Purpose:** Point to specific agent versions, enabling deployment strategies and rollback.

**Common Alias Patterns:**

| Alias Name | Points To | Purpose |
|------------|-----------|---------|
| **DRAFT** | Working Draft | Development and testing |
| **TEST** | v2 | Staging environment testing |
| **PROD** | v1 | Production traffic |
| **BETA** | v3 | Beta user testing |

**Benefits:**
- ✅ **Safe deployment** - Test new version before switching PROD alias
- ✅ **Instant rollback** - Change alias back to previous version
- ✅ **A/B testing** - Multiple aliases point to different versions
- ✅ **Environment separation** - DEV, TEST, PROD aliases

**Deployment Pattern:** `#exam-tip`

**Blue/Green Deployment:**
```
1. Current state:
   - PROD alias → v1 (stable)

2. Develop new features:
   - Edit Working Draft
   - Test with DRAFT alias

3. Create new version:
   - Create v2 from Working Draft
   - TEST alias → v2
   - Run integration tests

4. Deploy to production:
   - PROD alias → v2 (switch)
   - If issues: PROD alias → v1 (rollback)
```

**Exam Scenario:** `#exam-tip`
- **"Need to test agent changes before production"** → Create new version, use TEST alias
- **"Agent deployed but has issues, need to revert"** → Change PROD alias to previous version
- **"Run A/B test with different agent configurations"** → Use two aliases pointing to different versions

**Alias vs Version Comparison:**

| Aspect | Version | Alias |
|--------|---------|-------|
| **Mutability** | Immutable (cannot change) | Mutable (can point to different versions) |
| **Purpose** | Snapshot of agent config | Pointer to version for deployment |
| **Use Case** | Version control, audit trail | Environment management, rollback |
| **Creation** | Create from Working Draft | Point to any version |
| **Deletion** | Can delete old versions | Can delete aliases |

#### Agent + Knowledge Bases Integration `#important` `#exam-tip`

**Purpose:** Combine agent actions with RAG for information retrieval

**How It Works:** Agent queries Knowledge Base for information, then takes actions based on retrieved context

**Use Cases:**
- Customer support: Answers from docs + creates ticket
- HR assistant: Looks up policies + files request
- Sales agent: Checks product specs + generates quote

**Exam Tip:** Agent + Knowledge Base = RAG + Actions (information retrieval + task execution)

#### Agent Session State

**Purpose:** Maintain context across multi-turn conversations

**Features:** Session ID tracks conversation, agent remembers previous messages and context

#### Use Cases `#exam-tip`

**Common Patterns:**
- **Customer Service:** Handle requests with multiple actions (tickets, orders, refunds)
- **Travel Booking:** Multi-step itineraries (flights, hotels, cars)
- **IT Help Desk:** Knowledge Base (docs) + Action Groups (password resets, provisioning)
- **Sales Assistant:** Knowledge Base (products) + Action Groups (CRM, quotes)
- **HR Assistant:** Knowledge Base (policies) + Action Groups (leave, benefits)

#### Best Practices `#exam-tip`

- **Instructions:** Be specific about role, capabilities, and boundaries
- **Action Groups:** Single responsibility per group, clear descriptions, error handling
- **Versioning:** Test in DRAFT/TEST alias before PROD, keep 2-3 versions for rollback
- **Monitoring:** Track invocations in CloudWatch, alert on errors

#### Limitations `#exam-tip`

- **Execution time** - Agents can take longer than simple API calls (multi-step reasoning)
- **Cost** - Pay for foundation model tokens + Lambda invocations + Knowledge Base queries
- **Action Group limit** - Maximum number of Action Groups per agent
- **Orchestration control** - Agent decides actions (not deterministic like state machines)
- **Error handling** - Agent may not always handle errors gracefully

#### Agents vs Other AWS Services `#exam-tip`

| Capability | Bedrock Agents | Step Functions | Lambda | Lex |
|------------|----------------|----------------|--------|-----|
| **Natural language** | ✅ Yes | ❌ No | ❌ No | ✅ Yes (intents) |
| **Multi-step orchestration** | ✅ Autonomous | ✅ Explicit | ❌ Single function | ⚠️ Dialog management |
| **Reasoning** | ✅ LLM-powered | ❌ Pre-defined | ❌ No | ⚠️ Intent-based |
| **Tool calling** | ✅ Yes | ✅ Yes | N/A | ✅ Yes (fulfillment) |
| **Use case** | Conversational automation | Workflow orchestration | Single functions | Task-oriented bots |

**Key Exam Distinctions:**
- **Bedrock Agents** - Autonomous, LLM-powered, conversational workflows
- **Step Functions** - Deterministic, pre-defined workflows
- **Lex** - Intent-based bots (user explicitly states intent)
- **Lambda** - Single-purpose functions (not orchestration)

#### Exam Scenarios `#important` `#exam-tip`

| Scenario | Solution | Reasoning |
|----------|----------|-----------|
| "Build AI assistant that books hotels AND answers questions about travel policies" | **Bedrock Agent + Knowledge Base** | Need both RAG (policies) and actions (booking) |
| "Deploy new agent version without affecting production" | **Create version, use TEST alias** | Test before changing PROD alias |
| "Agent deployed has bug, need immediate fix" | **Change PROD alias to previous version** | Instant rollback |
| "Autonomous customer service that creates tickets and looks up orders" | **Bedrock Agent with Action Groups** | Multi-step, autonomous actions |
| "Want agent to call multiple Lambda functions based on user request" | **Bedrock Agent with multiple Action Groups** | Each Action Group = set of related functions |
| "Task-oriented bot with explicit intents (BookHotel, CheckWeather)" | **Lex** | Structured, intent-based (not autonomous agent) |
| "Complex workflow with error handling, retries, parallel steps" | **Step Functions** | Deterministic orchestration |
| "Agent needs to run A/B test with two different instruction sets" | **Create two versions, use two aliases** | Compare performance between versions |

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

**Common Bedrock Use Cases:**
- **Conversational AI:** Chatbots with Claude + RAG (no training needed)
- **Document Summarization:** Titan Text or Claude via API
- **Content Generation:** Marketing copy, product descriptions
- **Semantic Search:** Bedrock Embeddings + OpenSearch (meaning-based search)
- **Code Generation:** Claude for code assistance

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

- **Prompt Engineering:** Clear instructions, examples (few-shot learning), system prompts
- **Use RAG for Knowledge:** Cheaper than fine-tuning, provides current information
- **Implement Guardrails:** Filter PII, harmful content for customer-facing apps
- **Choose Right Model:** Start small (Titan, Claude Haiku), test in playground
- **Monitor Costs:** Track tokens, optimize prompts, use provisioned throughput for high volume

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
- [Amazon SageMaker](../sagemaker/sagemaker.md) - Custom ML model building
- [Cheat Sheet](../guides/cheat-sheet.md) - Quick reference
