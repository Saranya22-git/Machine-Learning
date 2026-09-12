Hey everybody!!!

# **Table of Contents**
- [**Table of Contents**](#table-of-contents)
- [**ML and Data Foundations**](#ml-and-data-foundations)
  - [**Machine Learning**](#machine-learning)
    - [**AI**](#ai)
    - [**Machine Learning**](#machine-learning-1)
    - [**Deep Learning**](#deep-learning)
    - [**AI vs ML vs DL**](#ai-vs-ml-vs-dl)
    - [**Traditional Programming vs Machine Learning**](#traditional-programming-vs-machine-learning)
    - [**Where ML is actually used?**](#where-ml-is-actually-used)
    - [**Types of ML problems**](#types-of-ml-problems)

# **ML and Data Foundations**

## **Machine Learning**

### **AI**

**What is AI?**

- *Artificial Intelligence (AI) is the field of creating computer systems that can perform tasks that normally require human-like intelligence.* 

**Examples:** *Understanding language, Recognizing images, Making decisions, Understanding speech, Recommending content, Playing games, Solving problems.*

---

**Example:** *Face Recognition*

*Your phone sees your face.*

```txt
Your face
    ↓
Camera
    ↓
AI system
    ↓
Recognizes face
    ↓
Unlocks phone
```

*The system is performing an intelligent recognition task.*

---

### **Machine Learning**

**What is Machine Learning?**

*Machine Learning is a subset of Artificial Intelligence where algorithms learn patterns from data and use those patterns to make predictions or decisions on new data.*

**Examples:** *Instead of manually writing thousands of rules, we give the ML algorithm examples.*

```txt
Email                         Label

"Win $1000 now!"              SPAM
"Claim your free prize"       SPAM
"Meeting at 10 AM"            NOT SPAM
"Project report attached"     NOT SPAM
```

*The ML algorithm looks at these examples and tries to discover patterns.*

```txt
Training Data
     ↓
ML Algorithm
     ↓
Learned Model
     ↓
New Email
     ↓
Prediction
     ↓
SPAM / NOT SPAM
```

---

**What does "Learn" mean?**

*The machine learns don't mean the computer kearns like a human.*

*The algorithm uses training data to estimate patterns or relationships and adjusts model parameters so that the model can make useful predictions on new data.*

*For example, during training, the model may discover that certain combinations of words, sender characteristics, message structure, etc. are associated with spam.*

*The model stores this learned information in its parameters.*

---

**ML Flow**

**Traditional Programming**

```txt
Rules + Data
     ↓
  Program
     ↓
  Output
```

**Machine Learning**

```txt
Data + Expected Outputs
          ↓
     ML Algorithm
          ↓
     Learned Model
```

*Then*

```txt
New Data
   ↓
Trained Model
   ↓
Prediction
```

*So the complete ML process is approximately*

```txt
Historical Data
      ↓
   Training
      ↓
 Trained Model
      ↓
   New Data
      ↓
   Prediction
```

---

**Traditional Programming vs Machine Learning**

| Traditional Programming         | Machine Learning                                         |
| ------------------------------- | -------------------------------------------------------- |
| Rules are explicitly programmed | Patterns are learned from data                           |
| Programmer defines logic        | Algorithm learns model parameters                        |
| Rules + data → output           | Data + examples → trained model                          |
| Works well when rules are clear | Useful when patterns are difficult to explicitly specify |
| Example: tax calculation        | Example: spam detection                                  |
| Example: even/odd checker       | Example: image classification                            |

---

**What are the main components of ML?**

```txt
Data
 ↓
Features
 ↓
Algorithm
 ↓
Model
 ↓
Prediction
```

*For example, customer churn*

```txt
Customer Data
     ↓
Age, tenure, monthly bill, contract
     ↓
ML Algorithm
     ↓
Trained Model
     ↓
Will customer leave?
     ↓
Yes / No
```

---

### **Deep Learning**

**What is Deep Learning?**

*Deep Learning is a subset of Machine learning that uses Neural networks with multiple layers to learn complex patterns and representations from data.*

---

**What is a Neural Network?**

*A neural network is a machine learning model made up of interconnected computational units called neurons.*

```txt
Input
  ↓
Neurons
  ↓
Neurons
  ↓
Output
```

*For example, suppose we want to predict whether an image contains a cat*

```txt
Image
  ↓
Neural Network
  ↓
CAT / NOT CAT
```

*The network processes the input and produces an output.*

---

**What does "Deep" mean?**

```txt
Input Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

```txt
Input Layer
     ↓
Hidden Layer 1
     ↓
Hidden Layer 2
     ↓
Hidden Layer 3
     ↓
Hidden Layer 4
     ↓
Output Layer
```

*The second network is deeper because it has more layers of computational.*

---

**Why Multiple Layers?**

*Imagine an image of a cat. An image is ultimately made up of pixel values. The network can learn increasingly complex representations.*

```txt
Pixels
  ↓
Edges
  ↓
Shapes
  ↓
Parts
  ↓
Object patterns
  ↓
CAT
```

```txt
Early layers
     ↓
Edges / simple patterns

Middle layers
     ↓
Shapes / textures

Later layers
     ↓
Eyes / ears / facial structures

Final layers
     ↓
Cat-like representation
```

---

**ML vs DL**

| Machine Learning                                              | Deep Learning                                       |
| ------------------------------------------------------------- | --------------------------------------------------- |
| Broader field within AI                                       | Subset of ML                                        |
| Includes many algorithms                                      | Primarily neural networks                           |
| Often works very well on tabular data                         | Particularly powerful for complex/unstructured data |
| Often requires manual feature engineering                     | Can learn representations automatically             |
| Can work with smaller datasets depending on algorithm/problem | Often benefits from large datasets                  |
| Usually less computationally expensive                        | Often computationally expensive                     |
| Examples: Logistic Regression, Random Forest, XGBoost         | Examples: CNNs, RNNs, Transformers                  |

---

**Relationship**

```txt
┌──────────────────────────────────────┐
│       ARTIFICIAL INTELLIGENCE        │
│                                      │
│   ┌──────────────────────────────┐   │
│   │     MACHINE LEARNING         │   │
│   │                              │   │
│   │   ┌──────────────────────┐   │   │
│   │   │   DEEP LEARNING      │   │   │
│   │   │                      │   │   │
│   │   │ Neural Networks      │   │   │
│   │   │ CNNs                 │   │   │
│   │   │ RNNs                 │   │   │
│   │   │ Transformers         │   │   │
│   │   └──────────────────────┘   │   │
│   │                              │   │
│   │ Logistic Regression          │   │
│   │ Decision Trees               │   │
│   │ Random Forest                │   │
│   │ XGBoost                      │   │
│   │ KNN                          │   │
│   │ SVM                          │   │
│   └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

---

### **AI vs ML vs DL**

```txt
Artificial Intelligence (AI)
│
├── Rule-based AI
│
└── Machine Learning (ML)
      │
      ├── Traditional ML
      │
      └── Deep Learning (DL)
```

*AI is the broadest concept. ML is a subset of AI. DL is a subset of ML.*

---

**The Relationship**

```txt
┌─────────────────────────────────────┐
│       ARTIFICIAL INTELLIGENCE       │
│                                     │
│    ┌───────────────────────────┐    │
│    │    MACHINE LEARNING       │    │
│    │                           │    │
│    │   ┌───────────────────┐   │    │
│    │   │  DEEP LEARNING    │   │    │
│    │   │                   │   │    │
│    │   │ Neural Networks   │   │    │
│    │   └───────────────────┘   │    │
│    └───────────────────────────┘    │
└─────────────────────────────────────┘
```

---

**Difference**

| Aspect              | AI                                      | Machine Learning            | Deep Learning                                       |
| ------------------- | --------------------------------------- | --------------------------- | --------------------------------------------------- |
| Meaning             | Broad field of intelligent systems      | Systems learn from data     | ML using deep neural networks                       |
| Scope               | Broadest                                | Narrower                    | More specialized                                    |
| Main idea           | Make machines perform intelligent tasks | Learn patterns from data    | Learn complex representations using multiple layers |
| Can use rules?      | Yes                                     | Can be part of an AI system | Primarily learned neural-network representations    |
| Data dependency     | Varies                                  | Generally data-driven       | Often highly data-driven                            |
| Feature engineering | Varies                                  | Often important             | Can learn representations automatically             |
| Typical examples    | Expert systems, intelligent agents      | Regression, trees, SVM, KNN | CNNs, RNNs, Transformers                            |

---

### **Traditional Programming vs Machine Learning**

**What is Traditional Programming?**

- *In traditional programming, a programmer explicitly writes the rules that tell the computer what to do.*
- *The basic flow is*
```txt
Rules + Data
     ↓
  Program
     ↓
   Output
```

*In ML, the situation is different. Instead of manually writing all those rules, we provide historical data.*

```txt
Data + Correct Answers
        ↓
    ML Algorithm
        ↓
    Learned Model
        ↓
      New Data
        ↓
     Prediction
```

---

**Example:** *Suppose we have this data*

| Hours Studied | Exam Result |
| ------------: | ----------- |
|             1 | Fail        |
|             2 | Fail        |
|             4 | Pass        |
|             5 | Pass        |
|             7 | Pass        |

*We want to predict whether a student will pass based on study hours.*

**Traditional programming:** *We could manually write*

```txt
IF hours >= 4
    → Pass
ELSE
    → Fail
```

**Machine Learning:** *Instead we give the historical data to an ML algorithm.*

```txt
Hours     Result
  1       Fail
  2       Fail
  4       Pass
  5       Pass
  7       Pass
```

- *The algorithm learns a relationship between study hours and the result.*
- *Then we give ```Hours = 6```*
- *The model might predict ```Pass```*
  
---

**Difference**

| Traditional Programming               | Machine Learning                              |
| ------------------------------------- | --------------------------------------------- |
| Programmer defines rules              | Algorithm learns patterns from data           |
| Rules are explicit                    | Model parameters/patterns are learned         |
| Data + Rules → Output                 | Data + Outputs → Model                        |
| Best when rules are known             | Useful when patterns are difficult to specify |
| Usually deterministic for same inputs | Predictions depend on learned model           |
| Example: tax calculation              | Example: churn prediction                     |
| Example: calculator                   | Example: spam detection                       |

---

### **Where ML is actually used?**

**What kinds of problems are suitable for ML?**

*ML is mainly useful when we have*

```txt
Data
 ↓
Patterns
 ↓
Prediction / Decision
```

*For example, "Based on a customer's past behavior, can we predict whether they will leave?"*

---

**Commom real-world ML applications**

**Customer Churn Prediction** 

**Business question:** *Which customers are likely to leave?*

**Data might contain:**

```txt
Customer age
Tenure
Contract type
Monthly charges
Usage
Support calls
```

*ML learn patterns from historical customers*

```txt
Customer Data
     ↓
ML Model
     ↓
Churn Probability
```

**Example:**

```txt
Customer A → 82% churn probability
Customer B → 12% churn probability
```

---

**Fraud Detection**

**Question:** *Is this transaction potentially fraudulent?*

*The model can learn from historical transactions*

```txt
Transaction
    ↓
ML Model
    ↓
Fraud Risk
```

**Potential Inputs:** *Transaction amount, Location, Time, Device, Transaction frequency, User behavior```*

**Example:**

```txt
Normal transaction → Low fraud risk
Unusual transaction → High fraud risk
```

---

**Recommendation Systems**

**Examples:** *Movies, Products, Music, Videos, News*

**Question:** *What is this user likely to be interested in?*

*For example*

```txt
Your previous activity
        ↓
ML model
        ↓
Learn preferences
        ↓
Recommendations
```

*If you repeatedly watch Python videos, a platform may recommend more programming-realted content.*

---

**Spam Detection**

**Question:** *Is this email/message spam?*

```txt
Email
 ↓
ML Model
 ↓
Spam / Not Spam
```

---

**Credit/Risk Prediction**

*For example, "What is the likelihood that a borrower will default?"*

**Possible data:**

```txt
Income
Credit history
Existing loans
Repayment history
Debt
```

*The model can estimate risk*

```txt
Applicant Data
      ↓
ML Model
      ↓
Risk Score / Probability
```

---

**Demand Forecasting**

**Business question:** *How much product might we need next week/month?*

*For example*

```txt
Historical sales
      ↓
ML / forecasting model
      ↓
Expected future demand
```

*Business can use this for Inventory planning, Staffing, Supply chain, Producing planning.*

---

**Healthcare Prediction**

**Examples:**
- *Disease-risk prediction*
- *Patient readmission prediction*
- *Medical image analysis*
- *Patient outcome prediction*

*For example*

```txt
Patient information
       ↓
ML Model
       ↓
Risk prediction
```

---

**Search and Ranking**

*ML can help determine "Which results should appear higher for a particular query or user?"*

```txt
Search Query
     ↓
Candidate Results
     ↓
ML Ranking
     ↓
Ordered Results
```

---

**Speech Recognition**

```txt
Voice
 ↓
ML / Deep Learning
 ↓
Text
```

*Used in Voice assistants, Transcription, Call-center systems, Accessibility tools.*

---

**Image Recognition**

**Question:** *What is present in this image?*

```txt
Image
 ↓
ML / Deep Learning
 ↓
Prediction
```

**Examples:**

- *Object detection*
- *Face-related applications*
- *Medical imaging*
- *Quality inspection*

---

*Machine Learning is used for problems such as Customer churn prediction, fraud detection, recommendation systems, spam detection, demand forecasting, risk prediction, and image or speech recognition. In these cases, historical data can be used to learn patterns and make predictions or decisions for new cases.*

---

```txt
Prediction
├── Churn
├── Fraud
├── Risk
├── Demand
└── Healthcare outcomes

Classification
├── Spam
├── Fraud / legitimate
└── Image categories

Recommendation
├── Products
├── Movies
├── Music
└── Content

Perception
├── Images
├── Speech
└── Video
```

---

### **Types of ML problems**