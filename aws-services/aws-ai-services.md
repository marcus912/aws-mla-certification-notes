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


## Quick Service Comparison `#important`

| Service | Category | Input | Output |
|---------|----------|-------|--------|
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

## When to Use What? `#exam-tip`

| Scenario | Service |
|----------|---------|
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

### Personalize vs SageMaker
- **Personalize:** Recommendations only, managed recipes
- **SageMaker:** Any ML problem, full flexibility

### Transcribe Medical vs Standard
- **Medical:** Healthcare terminology, HIPAA compliant
- **Standard:** General transcription

## Exam Tips Summary `#exam-tip`
- **No ML expertise needed:** All these services are pre-trained
- **Custom models available:** Comprehend, Rekognition, Personalize, Fraud Detector
- **Real-time + Batch:** Most services support both modes
- **Confidence scores:** Services return predictions with confidence
- **Integration:** All integrate with other AWS services (Lambda, S3, etc.)
- **Pricing:** Pay-per-use (per request, per minute, per character)

## Related Topics
- [AWS Generative AI & Foundation Models](./aws-generative-ai.md) - Bedrock, Amazon Q
- [Amazon SageMaker](../sagemaker/sagemaker.md) - Custom ML model building
- [Data Services](./data-services.md) - Data preparation and storage
- [Cheat Sheet](../guides/cheat-sheet.md) - Quick reference
