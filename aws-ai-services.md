# AWS AI Services

**Tags:** `#core` `#exam-tip` `#important`

## Overview
Pre-trained AI/ML services that don't require ML expertise. Ready-to-use APIs for common AI tasks.

## Natural Language Processing (NLP)

### Amazon Comprehend `#important`
**Purpose:** Text analysis and NLP

**Key Features:**
- **Sentiment analysis** - Positive, negative, neutral, mixed
- **Entity recognition** - People, places, dates, organizations, quantities
- **Key phrase extraction** - Important phrases from text
- **Language detection** - Identify language (100+ languages)
- **Topic modeling** - Discover topics in document collections
- **Syntax analysis** - POS tagging, tokenization
- **Custom entities** - Train custom entity recognizers
- **Custom classification** - Train custom classifiers
- **PII detection** - Identify personally identifiable information

**Use Cases:**
- Analyze customer feedback/reviews
- Content categorization
- Compliance (detect sensitive data)
- Social media sentiment monitoring

**Exam Tips:** `#exam-tip`
- Pre-trained service (no model training needed for basic features)
- Custom classification requires labeled training data
- Real-time (synchronous) and batch (asynchronous) modes
- Output: JSON with confidence scores

### Amazon Translate `#important`
**Purpose:** Neural machine translation

**Key Features:**
- 75+ languages supported
- Real-time and batch translation
- **Custom terminology** - Domain-specific translations
- **Active Custom Translation** - Fine-tune with parallel data
- **Formality setting** - Formal vs informal translation
- **Profanity masking**

**Use Cases:**
- Localize content for global users
- Translate customer support tickets
- Real-time chat translation
- Document translation

**Exam Tips:** `#exam-tip`
- Automatic source language detection
- Charged per character translated
- Supports plain text and HTML
- Custom terminology improves domain-specific accuracy

### Amazon Transcribe `#important`
**Purpose:** Speech-to-text (automatic speech recognition)

**Key Features:**
- Real-time and batch transcription
- **Speaker identification** (diarization) - "Who said what?"
- **Custom vocabulary** - Domain-specific terms, acronyms
- **Automatic punctuation and formatting**
- **PII redaction** - Remove sensitive audio/text
- **Language identification** - Auto-detect language
- **Multi-channel audio** - Separate channels (call center: agent vs customer)
- **Subtitle generation** - WebVTT, SRT formats

**Specialty Options:**
- **Medical** - Medical terminology (HIPAA eligible)
- **Call Analytics** - Sentiment, interruptions, talk time, issues detected

**Use Cases:**
- Meeting transcription
- Call center analytics
- Subtitle generation
- Voice assistant backends

**Exam Tips:** `#exam-tip`
- Supports 100+ languages
- Custom vocabulary improves accuracy for technical terms
- Call Analytics for customer service insights
- Medical version for healthcare (HIPAA compliant)

## Text-to-Speech

### Amazon Polly `#important`
**Purpose:** Text-to-speech synthesis

**Key Features:**
- **Neural TTS (NTTS)** - Most natural, human-like voices
- **Standard TTS** - Lower cost, less natural
- 60+ voices in 30+ languages
- **SSML support** - Control pronunciation, emphasis, pauses
- **Speech marks** - Metadata for lip-sync animation
- **Lexicons** - Custom pronunciation
- **Newscaster style** - News reading voice
- **Conversational style** - Natural conversation tone

**Use Cases:**
- Voice assistants
- E-learning narration
- Accessibility (screen readers)
- IVR systems (phone menus)

**Exam Tips:** `#exam-tip`
- Charged per character synthesized
- Neural voices more expensive but better quality
- Output formats: MP3, OGG, PCM
- Can stream audio for real-time playback

## Computer Vision

### Amazon Rekognition `#important`
**Purpose:** Image and video analysis

**Key Features (Images):**
- **Object and scene detection** - Cars, people, beaches, etc.
- **Facial analysis** - Age range, gender, emotions, attributes (glasses, beard)
- **Face comparison** - Compare two faces for similarity
- **Face search** - Search face in collection (face database)
- **Celebrity recognition** - Identify famous people
- **Text in images (OCR)** - Detect and extract text
- **Custom labels** - Train custom object/scene detection
- **Content moderation** - Detect inappropriate content (violence, nudity)
- **PPE detection** - Detect safety equipment (hard hats, masks)

**Key Features (Video):**
- All image features applied to video
- **Person tracking** - Track people across frames
- **Activity detection** - Running, dancing, etc.
- **Segment detection** - Black frames, shot changes, credits

**Use Cases:**
- Content moderation
- Facial authentication
- Person tracking in surveillance
- Product recognition
- Media analysis

**Exam Tips:** `#exam-tip`
- **Custom Labels** - AutoML for custom vision (no ML expertise)
- Minimum 10 images per label for custom models
- Real-time (images) and asynchronous (videos)
- Face collections for face search (up to 20M faces)
- Content moderation has confidence thresholds

### Amazon Textract `#exam-tip`
**Purpose:** OCR + document structure extraction

**Key Features:**
- **Text detection** - Like Rekognition but better for documents
- **Form extraction** - Key-value pairs (Name: John, DOB: 01/01/1990)
- **Table extraction** - Extract tables with relationships
- **Query-based extraction** - "What is the invoice number?"
- **Identity documents** - Passport, driver's license parsing
- **Invoice/receipt analysis** - Structured financial document data

**Difference from Rekognition:**
- **Rekognition:** Text in natural scenes (signs, labels)
- **Textract:** Document structure and forms (PDFs, scans)

**Use Cases:**
- Invoice processing automation
- Form digitization
- Document search and indexing
- KYC (Know Your Customer) verification

**Exam Tips:** `#exam-tip`
- Better than Rekognition for structured documents
- Returns bounding boxes and confidence scores
- Supports PDF, PNG, JPG, TIFF
- Queries feature for specific data extraction

## Conversational AI

### Amazon Lex `#important`
**Purpose:** Build conversational interfaces (chatbots, voice bots)

**Key Features:**
- **Intents** - User goals (BookHotel, CheckWeather)
- **Slots** - Required information (Date, Location, RoomType)
- **Fulfillment** - Execute action via Lambda
- **Multi-turn conversations** - Context maintenance
- **Sentiment analysis** - Built-in sentiment detection
- **Multi-language support**
- **8 kHz audio support** - Phone call quality

**Integration:**
- Website, mobile apps, Slack, Facebook Messenger
- Amazon Connect (call center)
- Polly for voice output, Transcribe for voice input

**Use Cases:**
- Customer service chatbots
- Voice assistants
- FAQ bots
- Transactional bots (order pizza, book appointment)

**Exam Tips:** `#exam-tip`
- Same tech as Amazon Alexa
- Lambda for business logic (fulfillment)
- Automatic speech recognition (ASR) and NLU built-in
- Versioning and aliases for bot deployment

## Search and Knowledge

### Amazon Kendra `#exam-tip`
**Purpose:** Intelligent enterprise search (NLP-powered)

**Key Features:**
- **Natural language queries** - "How do I reset my password?" (not just keywords)
- **ML-powered relevance** - Learns from user interactions
- **Connectors** - S3, SharePoint, Salesforce, ServiceNow, RDS, OneDrive
- **Document ranking** - Considers freshness, popularity, relevance
- **Incremental learning** - User feedback improves results
- **FAQ support** - Direct answers from FAQ documents
- **Access control** - Respects document permissions

**Use Cases:**
- Internal knowledge base search
- Customer support portals
- Research and investigation
- Compliance and regulatory search

**Exam Tips:** `#exam-tip`
- Enterprise search (not web search)
- Returns document excerpts with answers
- Understands natural language (better than Elasticsearch for NL queries)
- Expensive (Enterprise edition for high query volume)

## Personalization

### Amazon Personalize `#important`
**Purpose:** Real-time personalized recommendations

**Key Features:**
- **ML-powered recommendations** - Same tech as Amazon.com
- **Real-time** - Updates as user interacts
- **Recipes (algorithms):**
  - User personalization (recommendations for user)
  - Similar items (related products)
  - Personalized ranking (rerank items for user)
  - Trending items
- **Cold start** - Handles new users/items
- **Business rules** - Filter/promote certain items
- **A/B testing support**

**Input Data:**
- **Interactions** - User-item interactions (clicks, purchases, views)
- **Users** - User metadata (age, location)
- **Items** - Item metadata (category, price, description)

**Use Cases:**
- E-commerce product recommendations
- Content recommendations (movies, articles)
- Marketing personalization
- Email campaign personalization

**Exam Tips:** `#exam-tip`
- Requires historical interaction data (25+ users, 1000+ interactions minimum)
- Fully managed (no ML expertise needed)
- Real-time inference via API
- Recipes = pre-built algorithms for different use cases
- Incremental training with new data

## Human-in-the-Loop

### Amazon Augmented AI (A2I) `#exam-tip`
**Purpose:** Human review of ML predictions

**Key Features:**
- **Human review workflows** - Route low-confidence predictions to humans
- **Integration:** Rekognition, Textract, SageMaker, Custom models
- **Workforce options:**
  - Amazon Mechanical Turk (public crowdsourcing)
  - Private workforce (your employees)
  - Vendor workforce (third-party managed teams)
- **Review UI templates** - Custom or pre-built
- **Confidence thresholds** - Automatic triggering

**Use Cases:**
- Content moderation review
- Document verification (low confidence extractions)
- Sensitive data validation
- Quality assurance for ML predictions

**Workflow:**
1. ML model makes prediction
2. If confidence < threshold → route to human
3. Human reviews and corrects
4. Feedback can improve model

**Exam Tips:** `#exam-tip`
- Human-in-the-loop ML
- Use when high accuracy is critical
- Integrates with Rekognition (content moderation), Textract (forms)
- Can use for SageMaker custom models

### Amazon Mechanical Turk (MTurk) `#exam-tip`
**Purpose:** Crowdsourcing marketplace for human intelligence tasks (HITs)

**What It Is:**
- Platform connecting requesters with workers (called "Turkers")
- Workers perform tasks that require human judgment
- Pay-per-task model
- Global workforce (hundreds of thousands of workers)

**Key Characteristics:**
- **Public workforce** - Anyone can sign up as a worker
- **Scalable** - Handle thousands of tasks in parallel
- **Cost-effective** - Typically $0.01-$1.00 per task
- **Fast turnaround** - Tasks completed in minutes to hours
- **Quality control** - Qualification tests, worker ratings, majority voting

**Common ML Tasks:**
- **Data labeling** - Image classification, bounding boxes
- **Text annotation** - Sentiment labeling, entity tagging
- **Data collection** - Surveys, content generation
- **Data verification** - Validate ML predictions
- **Content moderation** - Flag inappropriate content

**Integration with AWS ML Services:**
- **SageMaker Ground Truth** - Uses MTurk for data labeling
- **Amazon A2I** - Uses MTurk for human review workflows
- **Direct API access** - Build custom HIT workflows

**When to Use MTurk vs Other Workforces:** `#exam-tip`

| Use Case | Workforce Choice |
|----------|------------------|
| Public data, low sensitivity | Mechanical Turk (cheapest, fastest) |
| Confidential/proprietary data | Private workforce (your employees) |
| Need domain expertise | Vendor workforce (managed experts) |
| High volume, simple tasks | Mechanical Turk |
| Complex tasks requiring training | Private or vendor workforce |
| HIPAA/PII data | Private workforce (NOT MTurk) `#gotcha` |

**Exam Tips:** `#exam-tip`
- **MTurk = public workforce** - Don't use for sensitive/confidential data
- **Ground Truth uses MTurk** by default but also supports private workforces
- **A2I workforce options:** MTurk (public), Private (your team), Vendor (managed)
- **Cost:** MTurk cheapest, Vendor most expensive
- **Quality control:** Use consensus (multiple workers), qualifications, gold standards
- **Not for sensitive data:** PHI, PII, trade secrets → use private workforce `#gotcha`

**Quality Control Strategies:**
1. **Consensus** - Multiple workers label same item, use majority vote
2. **Qualifications** - Require workers to pass tests
3. **Master workers** - Pre-vetted high-quality workers (premium)
4. **Gold standard data** - Hidden test items with known answers
5. **Worker ratings** - Track and filter by accuracy

**Pricing Model:**
- Pay per completed task (HIT)
- Set your own price per task
- MTurk takes 20% fee (40% for 10+ workers per task)
- Example: $0.10/task, 1000 tasks = $120 total cost ($100 + $20 fee)

## Anomaly Detection

### Amazon Lookout Family `#exam-tip`

**Purpose:** Industry-specific anomaly detection services

**Services:**

#### Lookout for Metrics
- **Purpose:** Detect anomalies in business metrics (KPIs)
- **Use case:** Revenue drops, traffic spikes, conversion rate changes
- **Features:**
  - Automatic anomaly detection
  - Root cause analysis
  - Integrations: S3, Redshift, CloudWatch, AppFlow (SaaS sources)

#### Lookout for Vision
- **Purpose:** Detect defects in manufacturing (visual inspection)
- **Use case:** Quality control, product inspection
- **Features:** Computer vision for defect detection
- **Training:** Only 30 images needed

#### Lookout for Equipment
- **Purpose:** Detect equipment anomalies (predictive maintenance)
- **Use case:** Industrial equipment, IoT sensors
- **Features:** Sensor data analysis, early failure detection

**Exam Tips:** `#exam-tip`
- **Metrics:** Business KPIs
- **Vision:** Manufacturing defects
- **Equipment:** Industrial IoT sensors
- All use ML for anomaly detection without ML expertise

## Fraud Detection

### Amazon Fraud Detector `#important`
**Purpose:** Detect online fraud using ML

**Key Features:**
- **Pre-built fraud models:**
  - Online fraud (account takeover, payment fraud)
  - Account takeover prevention
  - Guest checkout fraud
- **Custom fraud models** - Train on your fraud data
- **Rules engine** - Combine ML + business rules
- **Real-time scoring** - Fraud risk score per event
- **Explainability** - Top fraud indicators

**Input Data:**
- Transaction data
- Account data
- Customer behavior

**Use Cases:**
- E-commerce fraud prevention
- Account registration fraud
- Payment fraud detection
- Loyalty program abuse

**Exam Tips:** `#exam-tip`
- Pre-built models (no training data needed)
- Custom models need historical fraud labels
- Real-time API for transaction scoring
- Combine ML scores with business rules

## Generative AI & Foundation Models `#important`

### Amazon Bedrock `#exam-tip`

**Purpose:** Managed service to access and use foundation models (FMs) via API

**Key Concept:** Pre-trained large models (LLMs, image generation) without managing infrastructure or training models from scratch.

#### What Is Bedrock?

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

#### Available Foundation Models `#important`

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

#### Key Features

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

#### Bedrock vs SageMaker Decision Framework `#important` `#exam-tip`

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

#### Use Cases `#exam-tip`

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

#### Integration with AWS Services

**Bedrock integrates with:**
- **S3** - Store documents for RAG
- **Lambda** - Serverless application backends
- **OpenSearch Serverless** - Vector search for RAG
- **SageMaker** - Can combine Bedrock FMs with custom SageMaker models
- **CloudWatch** - Monitoring and logging
- **IAM** - Access control
- **VPC** - Private network access

#### Pricing `#exam-tip`

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

#### Best Practices `#exam-tip`

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

#### Limitations `#exam-tip`

- **Token limits** - Context window varies by model (4K-200K tokens)
- **No fine-grained control** - Can't modify model architecture
- **Model availability** - Limited to available foundation models
- **Not for all ML tasks** - Primarily for generative AI (text, images)
- **Cost at scale** - Token-based pricing can be expensive at very high volumes

#### Exam Scenarios `#exam-tip`

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

## Generative AI Assistant (Q Family)

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

## Quick Service Comparison `#important`

| Service | Category | Input | Output |
|---------|----------|-------|--------|
| **Bedrock** | **Generative AI / Foundation Models** | **Text/Images** | **Generated text, images, embeddings** |
| Comprehend | NLP | Text | Sentiment, entities, topics |
| Translate | NLP | Text | Translated text |
| Transcribe | Speech | Audio | Text transcript |
| Polly | Speech | Text | Audio speech |
| Rekognition | Vision | Image/Video | Objects, faces, text, moderation |
| Textract | Vision | Documents | Structured text, forms, tables |
| Lex | Conversational | Text/Voice | Intent, slots, response |
| Kendra | Search | Query + Documents | Relevant documents, answers |
| Personalize | Recommendations | User interactions | Personalized recommendations |
| A2I | Human Review | ML predictions | Human-validated results |
| Lookout | Anomaly Detection | Metrics/Images/Sensors | Anomalies |
| Fraud Detector | Fraud | Transaction data | Fraud risk score |
| Q Business | Generative AI | Enterprise data | Answers, summaries |
| Q Developer | Generative AI | Code context | Code suggestions |

## When to Use What? `#exam-tip`

| Scenario | Service |
|----------|---------|
| **Build chatbot with conversational AI** | **Bedrock (Claude) or Lex** |
| **Summarize documents automatically** | **Bedrock** |
| **Generate marketing copy, content** | **Bedrock** |
| **Generate images from text** | **Bedrock (Stable Diffusion, Titan Image)** |
| **Semantic search (meaning-based)** | **Bedrock Embeddings + OpenSearch** |
| Analyze customer reviews sentiment | Comprehend |
| Translate website to 10 languages | Translate |
| Transcribe meeting recordings | Transcribe |
| Generate voiceovers for videos | Polly |
| Moderate user-uploaded images | Rekognition |
| Extract data from invoices | Textract |
| Build task-oriented chatbot (intents/slots) | Lex |
| Search company knowledge base (NLP) | Kendra |
| Recommend products to users | Personalize |
| Human review of low-confidence predictions | Augmented AI (A2I) |
| Detect revenue anomalies | Lookout for Metrics |
| Detect defects in manufacturing | Lookout for Vision |
| Detect fraudulent transactions | Fraud Detector |
| Q&A over company documents | Q Business or Bedrock + RAG |
| Code completion assistance | Q Developer |
| Detect text in street signs | Rekognition |
| Extract form fields from PDFs | Textract |
| Track person across video | Rekognition Video |

## Key Differences `#exam-tip`

### Rekognition vs Textract
- **Rekognition:** Natural scenes, general OCR, faces, objects
- **Textract:** Documents, forms, tables, structured data

### Kendra vs Elasticsearch/OpenSearch
- **Kendra:** Natural language understanding, ML-powered relevance
- **Elasticsearch:** Keyword search, full-text search

### Comprehend vs SageMaker
- **Comprehend:** Pre-trained NLP, no ML expertise
- **SageMaker:** Custom models, full ML control

### Bedrock vs SageMaker `#important`
- **Bedrock:** Pre-trained foundation models, generative AI (text, images), no training needed
- **SageMaker:** Custom ML models, any algorithm, train from scratch

### Bedrock vs Lex
- **Bedrock:** Conversational AI with LLMs, open-ended conversations, RAG
- **Lex:** Task-oriented bots with intents/slots, structured workflows

### Bedrock vs Q Business
- **Bedrock:** Foundation model platform (API), build your own apps
- **Q Business:** Pre-built generative AI assistant, ready to use

### Personalize vs SageMaker
- **Personalize:** Recommendations only, managed recipes
- **SageMaker:** Any ML problem, full flexibility

### Transcribe Medical vs Standard
- **Medical:** Healthcare terminology, HIPAA compliant
- **Standard:** General transcription

## Exam Tips Summary `#exam-tip`
- **No ML expertise needed:** All these services are pre-trained
- **Bedrock for generative AI:** Use for text generation, chatbots, summarization, image generation
- **SageMaker for custom ML:** Use for traditional ML (classification, forecasting, custom models)
- **Custom models available:** Bedrock (fine-tuning), Comprehend, Rekognition, Personalize, Fraud Detector
- **Real-time + Batch:** Most services support both modes
- **Confidence scores:** Services return predictions with confidence
- **Integration:** All integrate with other AWS services (Lambda, S3, etc.)
- **Pricing:** Pay-per-use (per request, per minute, per character, per token for Bedrock)

## Related Topics
- [Amazon SageMaker](./sagemaker.md)
- [Data Services](./data-services.md)
- [Cheat Sheet](./cheat-sheet.md)
