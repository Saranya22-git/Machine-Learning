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
      - [**Supervised Learning**](#supervised-learning)
        - [**Classification**](#classification)
        - [**Regression**](#regression)
      - [**Unsupervised Learning**](#unsupervised-learning)
        - [**Clustering**](#clustering)
        - [**Dimensionality Reduction**](#dimensionality-reduction)
        - [**Anomaly Detection**](#anomaly-detection)
      - [**Semi-Supervised Learning**](#semi-supervised-learning)
      - [**Self-Supervised Learning**](#self-supervised-learning)
      - [**Reinforcement Learning**](#reinforcement-learning)
      - [**Supervised vs Unsupervised**](#supervised-vs-unsupervised)
    - [**Real-World ML workflow**](#real-world-ml-workflow)
    - [**When should you use ML?**](#when-should-you-use-ml)
    - [**When should you NOT use ML?**](#when-should-you-not-use-ml)
  - [**ML Terminology**](#ml-terminology)
    - [**Dataset**](#dataset)
    - [**Feature**](#feature)
    - [**Target / Label**](#target--label)
    - [**Observation / Instance**](#observation--instance)
    - [**Training / Validation / Test Data**](#training--validation--test-data)
      - [**Training Data**](#training-data)
      - [**Validation Data**](#validation-data)
      - [**Test Data**](#test-data)
    - [**Model**](#model)
    - [**Parameters**](#parameters)
    - [**Hyperparameters**](#hyperparameters)
    - [**Prediction**](#prediction)
    - [**Inference**](#inference)
    - [**Training**](#training)
    - [**Evaluation**](#evaluation)
    - [**Loss**](#loss)

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

**What is an ML problem type?**

*An ML problem type describes what the model is expected to learn and produce.*

*For example ```"Will this customer leave?"``` The answer is ```Yes / No``` That's a classification problem.*

*But ```"What will this house cost?"``` The answer is a numerical value ```₹75 lakh``` That's a regression problem.*

---

**Main Types**

```txt
Machine Learning Problems
│
├── Supervised Learning
│     │
│     ├── Classification
│     └── Regression
│
└── Unsupervised Learning
      │
      ├── Clustering
      ├── Dimensionality Reduction
      └── Anomaly Detection
```

*There are also*

```txt
Semi-supervised Learning
Self-supervised Learning
Reinforcement Learning
```

---

#### **Supervised Learning**

*Supervised learning is a type of machine learning where the model learns from labeled training data, meaning the desired output is provided during training.*

```txt
Input + Correct Output
        ↓
      Model
        ↓
Learn relationship
        ↓
New Input
        ↓
Prediction
```

**Example:** *Suppose we have*

| Hours Studied | Result |
| ------------- | ------ |
|             2 | Fail   |
|             4 | Fail   |
|             6 | Pass   |
|             8 | Pass   |

*The model sees both*

```txt
Input → Hours studied
Output → Pass/Fail
```

*The output is already known in the training data. That's why it's called ```Supervised```*

---

**Two major Supervised Learning Problems**

```txt
Supervised Learning
       │
       ├── Classification
       │
       └── Regression
```

**Classification:** *Output is a category/class.*

**Regression:** *Output is a numerical value.*

---

##### **Classification**

*Classification is a supervised learning problem where the model predicts a discrete class or category.*

**Examples:**

```txt
Spam / Not Spam
Fraud / Legitimate
Pass / Fail
Disease / No Disease
Churn / No Churn
```

*The output belongs to a category.*

---

**Example for Classification:** *Customer Churn*

**Question:** *Will the customer leave?*

**Possible output:** *Yes/No*

```txt
Customer Data
     ↓
ML Model
     ↓
Yes / No
```

---

**Binary Classification:** *If there are two possible classes, it's called binary classification.*

**Examples:**

```txt
Spam / Not Spam
Yes / No
Fraud / Not Fraud
Churn / No Churn
Pass / Fail
```

---

**Multiclass Classification:** *What if there are more than two categories?*

*For example*

```txt
Email category:
    ↓
Work
Personal
Promotion
Social
Spam
```

**Example:**

```txt
Image
 ↓
Cat / Dog / Horse
```

*There are multiple possible classes*

---

##### **Regression**

*Regression is a supervised learning problem where the model predicts a continuous numerical value.*

**Examples:**

```txt
House price
Salary
Temperature
Sales
Revenue
Demand
```

*For example*

```txt
House features
      ↓
ML Model
      ↓
₹82,50,000
```

*The output is a number.*

---

**Classification vs Regression**

| Classification            | Regression                     |
| ------------------------- | ------------------------------ |
| Predicts a class/category | Predicts a numerical value     |
| Output is discrete        | Output is generally continuous |
| Example: Spam/Not Spam    | Example: House price           |
| Example: Churn/No Churn   | Example: Sales amount          |
| Example: Fraud/Legitimate | Example: Temperature           |

---

#### **Unsupervised Learning**

*Unsupervised learning works with data where the desired target labels are not provided, and the goal is to discover useful structure or patterns in the data.*

---

```txt
Data
 ↓
Algorithm
 ↓
Discover structure/patterns
```

**Example:** *Customer Segmentation*

*Suppose a company has customer information*

```txt
Age
Income
Purchase frequency
Spending
```

*But there is no label saying*

```txt
Customer A → Segment 1
Customer B → Segment 2
```

*We can ask the algorithm "Can you find groups or similar customers?"*

*The model might discover*

```txt
Group 1 → High spending
Group 2 → Low spending
Group 3 → Frequent buyers
```

*This is unsupervised learning.*

---

**Main Unsupervised Types**

```txt
Unsupervised Learning
│
├── Clustering
├── Dimensionality Reduction
└── Anomaly Detection
```

---

##### **Clustering**

*Group similar data points together.*

**Example:**

```txt
Customers
    ↓
Clustering
    ↓
Customer Groups
```

**Common example:** *Customer segmentation*

---

##### **Dimensionality Reduction**

*Reduce the number of features while trying to preserve important information.*

*Suppose we have ```100 features``` We may want a representation with ```10 dimensions``` while retaining as much useful information as possible.*

*One important technique is ```PCA - Principal Component Analysis```*

---

##### **Anomaly Detection**

*Identify observations that are unusual compared with normal patterns.*

**Example:**

```txt
Normal transactions
Normal transactions
Normal transactions
        ↓
Unusual transaction
        ↓
Potential anomaly
```

*Applications include Fraud detection, Network monitoring, Equipment monitoring.*

---

#### **Semi-Supervised Learning**

*Semi-supervised learning combines labeled and unlabeled data during learning, which can be useful when labeled data is limited but unlabeled data is abundant.*

*You have*

```txt
Small amount of labeled data
+
Large amount of unlabeled data
```

**Example:**

```txt
1,000 labeled images
+
100,000 unlabeled images
```

*The learning approach uses both*

---

#### **Self-Supervised Learning**

*The system creates a learning signal from the data itself rather than requiring humans to manually provide every label.*

*For example, a model may be given part of sentence and trained to predict a missing/next part.*

```txt
Text:
"The cat is sitting on the ___"

             ↓

Model predicts:
"mat"
```

*The training signal comes from the original data itself*

---

#### **Reinforcement Learning**

*Reinforcement Learning is a type of learning where an agent interacts with an environment and learns actions through rewards and penalties.*

```txt
Agent
  ↓
Action
  ↓
Environment
  ↓
Reward / Penalty
  ↓
Learn better actions
```

**Example:** *A game-playing agent*

```txt
Move
 ↓
Game state
 ↓
Win → Reward
Lose → Penalty
```

*The agent learns a strategy through interaction.*

---

#### **Supervised vs Unsupervised**

| Supervised                              | Unsupervised                    |
| --------------------------------------- | ------------------------------- |
| Uses labeled data                       | Uses data without target labels |
| Has known target/output during training | Target isn't provided           |
| Learns input → output relationship      | Finds patterns/structure        |
| Classification                          | Clustering                      |
| Regression                              | Dimensionality reduction        |
| Example: churn prediction               | Example: customer segmentation  |

---

*At a high level, machine-learning problems can be supervised or unsupervised. In supervised learning, the model learns from labeled data and common tasks are classification and regression. In unsupervised learning, there is no predefined target and we try to discover structure, such as through clustering, dimensionality reduction, or anomaly detection. There are also approaches such as semi-supervised, self-supervised, and reinforcement learning.*

---

### **Real-World ML workflow**

**What is an ML workflow?**

*An ML workflow is the sequence of steps followed to take a real-world problem from*

```txt
Business Problem
      ↓
Data
      ↓
Model
      ↓
Evaluation
      ↓
Deployment
      ↓
Monitoring
```

---

**Complete Real-World Workflow**

```txt
1. Define Business Problem
          ↓
2. Collect Data
          ↓
3. Understand / Explore Data
          ↓
4. Prepare Data
          ↓
5. Split Data
          ↓
6. Train Model
          ↓
7. Evaluate Model
          ↓
8. Improve / Select Model
          ↓
9. Deploy
          ↓
10. Monitor & Maintain
```

---

- **Step-1:** *Define the Business Problem*

     *Suppose a company says "Our customers are leaving". That's a business problem. You need to translate it into an ML problem: "Can we predict which customers are likely to churn?"*

     ```txt
     Business Problem
          ↓
     ML Objective
          ↓
     Target
          ↓
     Features
          ↓
     Prediction
     ```

     *For example*

     ```txt
     Business objective: Reduce customer churn

     ML objective: Predict customers likely to churn

     Target:Churn / No Churn
     ```

---

- **Step-2:** *Collect data*

     *Once we know what we're trying to predict, we need relevant data. For churn prediction we night collect*

     ```txt
     Customer ID
     Age
     Tenure
     Contract
     Monthly Charges
     Usage
     Support Calls
     Payment History
     Churn
     ```

     *Data can come from different sources*
     - *Databases*
     - *APIs*
     - *Application logs*
     - *Business systems*
     - *Files*
     - *Sensors*
     - *Existing datasets*

     *The important question is "Do we have data that contains information useful for solving the problem?"*

---

- **Step-3:** *Understand the Data*

     *Don't immediately train a model. First understand what you've collected. Ask*

     ```txt
     How many rows?
     How many columns?
     What does each column mean?
     Which column is the target?
     Are values missing?
     Are there incorrect values?
     Are there categorical variables?
     Are there unusual values?
     ```

     *For example*

     ```txt
     Age → numerical
     Monthly Charges → numerical
     Contract → categorical
     Churn → target
     ```

---

- **Step-4:** *Prepare the data*

     *Real-world data is rarely ready to directly feed into a model. You may encounter*

     ```txt
     Missing values
     Incorrect values
     Duplicate records
     Outliers
     Categorical variables
     Different scales
     ```

     *So we may need to perform preprocessing*

     *For example*

     ```txt
     Raw Data
     ↓
     Handle missing values
     ↓
     Handle categorical data
     ↓
     Scale / transform when appropriate
     ↓
     Clean dataset
     ```

---

- **Step-5:** *Split the data*

     *We generally don't want to train and evaluate on exactly the same data. Because we want to know "Can the model perform well on data it hasn't seen during training?". So we typically divide data into subsets such as*

     ```txt
     Dataset
     │
     ├── Training Data
     ├── Validation Data
     └── Test Data
     ```

     **Training Data:** *Used to train the model*

     **Validation Data:** *Used during model selection/tuning*

     **Test Data:** *Used for final evaluation*

     *We need unseen data to assess how well the model generalizes.*

---

- **Step-6:** *Train the model*

     *Now we select an appropriate algorithm. For example*

     ```txt
     Classification problem
          ↓
     Possible models
     ├── Logistic Regression
     ├── Decision Tree
     ├── Random Forest
     └── Gradient Boosting
     ```

     *We train the model using the training data*

     ```txt
     Training Data
          ↓
     Algorithm
          ↓
     Training
          ↓
     Trained Model
     ```

     *The model learns parameters from the training data*

---

- **Step-7:** *Evaluate the Model*

     *This is where we ask "How good is the model?". Suppose our model predicts churn. We can't simply say "It looks good". We need appropraite evaluation metrics.*

     *For classification, examples include*
     - *Accuracy*
     - *Precision*
     - *Recall*
     - *F1-score*
     - *ROC-AUC*

     *For regression*
     - *MAE*
     - *MSE*
     - *RMSE*
     - *R²*

     *The metric should match the business problem*

---

- **Step-8:** *Improve / Select the Model*

     *Suppose we test*

     ```txt
     Logistic Regression → 0.82 AUC
     Random Forest      → 0.87 AUC
     Gradient Boosting  → 0.90 AUC
     ```

     *We don't simply choose the largest number blindly*

     *We consider*
     - *Performance*
     - *Interpretability*
     - *Training cost*
     - *Prediction speed*
     - *Data requirements*
     - *Business requirements*
     - *Deployment constraints*

     *Then select a model that best fits the overall requirements*

---

- **Step-9:** *Deploy the model*

     *A trained model sitting inside a notebook isn't necessarily useful to the business. We need to make it available to an application or system.*

     *For example*

     ```txt
     User / Application
          ↓
          API
          ↓
     ML Model
          ↓
     Prediction
     ```

     **Example:**

     ```txt
     Customer data
          ↓
     API
          ↓
     Churn model
          ↓
     82% churn probability
     ```

     *This is where topics such as*
     - *Flask/FastAPI*
     - *Docker*
     - *Cloud*
     - *Model serving*

     *can become relevant*

---

- **Step-10:** *Monitor the model*

     *You deploy the model today. Does that mean you're finished? ```No``` Real-world data can change*

     *For example*

     ```txt
     Training data
          ↓
     2025 customer behavior
     ```

     *But after deployment ```2026 customer behavior``` may be different*

     *The model may become less accurate. So we need monitoring*

     *Things we may monitor include*
     - *Prediction performance*
     - *Data quality*
     - *Input distributions*
     - *Model latency*
     - *Errors*
     - *Data drift*
     - *Model drift*

---

**Real-World Example - Complete Workflow**

*A telecom company wants to reduce customer churn*

- **Step-1:** *Business problem ```Customers are leaving.```*
- **Step-2:** *ML problem ```Predict customers likely to churn.```*
- **Step-3:** *Data*
     ```txt
     Customer history
          ↓
     Usage
     Contract
     Charges
     Support interactions
     Churn history
     ```
- **Step-4:** *EDA*
     *Understand*
     ```txt
     Missing values
     Distributions
     Relationships
     Class balance
     ```
- **Step-5:** *Preprocessing*
     ```txt
     Clean data
     Encode categories
     Handle missing values
     etc.
     ```
- **Step-6:** *Split ```Train / Validation / Test```*
- **Step-7:** *Train*
     *Try suitable models*
     ```txt
     Logistic Regression
     Random Forest
     Gradient Boosting
     ```
- **Step-8:** *Evaluate ```Use appropriate classification metrics*
- **Step-9:** *Select ```Choose the model based on both performance and practical requirements*
- **Step-10:** *Deploy ```Application → API → Model → Prediction```*
- **Step-11:** *Monitor*
     ```txt
     Production Data
          ↓
     Monitor performance
          ↓
     Detect changes
          ↓
     Retrain / update if necessary
     ```

---

**ML Workflow is NOT Linear Forever**

*The diagram looks like*

```txt
Problem
 ↓
Data
 ↓
Model
 ↓
Deploy
```

*But in reality, you'll often go backward*

*For example*

```txt
Train Model
    ↓
Poor performance
    ↓
Investigate data
    ↓
Improve preprocessing/features
    ↓
Train again
```

*So it's more like*

```txt
        ┌──────────────┐
        ↓              │
Data → Train → Evaluate
        ↑              │
        └── Improve ←──┘
```

*ML development is usually iterative*

---

**What are the steps in an ML project?**

*First, I would understand and define the business problem and translate it into an ML objective. Then I would collect and understand the relevant data, perform EDA and preprocessing, split the data appropriately, train candidate models, evaluate them using metrics relevant to the business problem, select and tune the best suitable model, deploy it, and finally monitor its performance in production.*

---

### **When should you use ML?**

*Don't start with "Can ML solve this?". Start with "Does this problem actually need ML?"*

*A good ML use case generally has*

```txt
A meaningful prediction/decision problem
                +
Relevant historical data
                +
Patterns that are difficult to write as fixed rules
                +
Enough business value to justify ML
```

---

**Condition 1 - There is a Prediction or Decision Problem**

*ML is useful when we want the system to predict something or make a data-driven decision.*

**Examples:** 

- *Customer Churn - Which customers are likely to leave?*

     ```txt
     Customer data
          ↓
     ML
          ↓
     Churn prediction
     ```

- *House Price - What will this house likely sell for?*

     ```txt
     House information
          ↓
     ML
          ↓
     Predicted price
     ```

- *Fraud - Is this transaction potentially fraudulent?*

```txt
Transaction data
      ↓
ML
      ↓
Fraud risk
```

*These are natural ML problems*

---

**Condition 2 - You have Relevant Data**

*Suppose a company says "We want to predict which customers will churn". You should ask "Do we have historical customer data?"*

*For example*

```txt
Customer history
     ↓
Past behavior
     ↓
Past churn outcomes
```

*If we have useful historical examples, ML has something to learn from*

---

**What if there is no useful data?**

- *Suppose "We launched a completely new product yesterday and have no historical customer behavior".* 
- *It may be difficult to train a useful supervised model for that specific prediction because there may not be enough relevant historical data*
- *So having a problem that sounds like ML does not automatically mean you can successfully use ML*
- *Data availability matters*

---

**Condition 3 - There are Patterns we cannot easily write as Rules**

*Consider Calculate GST. The rules/formula are known*

```txt
Known formula
     ↓
Traditional Program
```

*No need for ML*

*Now consider Detect spam emails. There can be huge numbers of different patterns*

```txt
Emails
  ↓
Complex patterns
  ↓
ML
```

*Manually writing every possible spam rule would be difficult.*

---

**Condition 4 - The Pattern is Learnable from Data**

*Having data isn't enough. The data needs to contain useful information related to the target.*

*Suppose you want to predict "Will this customer churn?"*

*But your dataset contains only*

```txt
Customer ID
Name
Favorite color
```

*There may not be useful predictive information*

*So*

```txt
Data available
      ≠
Useful data available
```

*You need features that contain meaningful signals for the prediction task*

---

**Condition 5 - ML provides enough value**

*ML introduces additional complexity*

*You may need*
- *Data pipelines*
- *Model training*
- *Evaluation*
- *Deployment*
- *Monitoring*
- *Retraining*

*So ask "Is the benefit of using ML worth the additional complexity?"*

*Suppose a simple rule gives exactly the required result*

```txt
IF balance < 0 → Send alert
```

*Building a neural network for this would be unnecessary*

---

**ML vs Rule-Based System**

*Use tarditional programming when*

```txt
Rules are known
      +
Rules are stable
      +
Exact behavior is required
```

**Example:** *Tax calculation, Eligibility rule, Billing formula, Password validation*

*Consider ML when*

```txt
Patterns are complex
      +
Relevant data exists
      +
Prediction is needed
      +
Rules are difficult to manually define
```

**Example:** *Fraud detection, Churn prediction, Recommendation, Spam detection, Image classification*

---

### **When should you NOT use ML?**

1. *Use a simple rule when the rule is already known*

     *Suppose If the customer's balance is below ₹0 send an alert. The rule is already known*

     ```txt
     IF balance < 0
          ↓
     Send Alert
     ```

     *There is nothing to learn. Use Traditional Programming*

2. *Use a formula when the relationship is known*

     *Suppose you need to calculate simple interest ```SI = P × R × T / 100``` There is no reason to train an ML model. Use a normal program*


3. *When there is no useful data*

     *Suppose a company asks "Predict which customers will buy our new product". But the company has*

     ```txt
     No historical customer data
     No previous purchases
     No relevant behavior data
     ```

     *There may not be enough information for a useful data-driven model*

4. *When the Data doesn't contain useful signals*

     *Suppose you want to predict "Will this customer churn?" But your dataset contains only*

     ```txt
     Customer ID
     Name
     Favorite Color
     ```

     *Even if you have ```10 million customers``` those columns may not provide meaningful information for churn prediction*

     *So more data does not automatically mean better ML. The data needs relevant predictive information*

5. *When the Problem requires exact deterministic behavior*

     *Suppose a banking system has a rule*

     ```txt
     If account balance < ₹0
     → Account is overdrawn
     ```

     *The system should behave predictably. You don't want*

     ```txt
     Model says: "Probably overdrawn: 96%"
     ```

     *When the rule is already exact. Use deterministic programming*

6. *When ML adds unnecessary complexity*

     *Imagine a company wants ```Convert Celsius to Fahrenheit```*

     **Formula:** *```F = (C × 9/5) + 32```*

     *You could train a model. But why?*

     ```txt
     Formula
     ↓
     Exact answer
     ```

     *is much simpler than*

     ```txt
     Collect data
     ↓
     Train model
     ↓
     Evaluate
     ↓
     Deploy
     ↓
     Monitor
     ↓
     Maintain
     ```

     *Don't use ML when a simpler solution already solves the problem reliably*

7. *When the cost of ML isn't justified*

     *ML systems have costs*
     - *Data collection*
     - *Data storage*
     - *Training*
     - *Infrastructure*
     - *Deployment*
     - *Monitoring*
     - *Maintenance*
     - *Retraining*

     *If the business benefit is tiny, building an ML system may not be worthwhile*

     *For example, A company needs to automatically classify only 20 records per month using a simple rule. A complicated ML pipeline would probably be unnecessary*

8. *When Explainability is critical and a simple rule works*

     *Sometimes the business requires highly understandable decisions*

     *Suppose*

     ```txt
     IF age >= 18
     AND required document = present
     → Eligible
     ```

     *The decision is immediately understandable. If an ML model isn't needed to solve the problem, using a complex model could make the system harder to explain*

     *This doesn't mean ML can never be used where explainability matters. It means that if a simple transparent approach already meets the requirement it may be preferable.*

9.  *When the Problem is better solved by another method*

     *Not every data problem is necessarily an ML problem. Depending on the problem other approaches may be better*

     ```txt
     Known formula       → Traditional programming
     Database lookup     → SQL / database query
     Business rule       → Rule-based system
     Optimization        → Optimization algorithms
     Simple statistics   → Statistical method
     ML needed           → Machine learning
     ```

     *The goal isn't "Use ML". The goal is "Choose the appropriate solution"*

---

```txt
                Problem
                   ↓
          Is the rule already known?
              /            \
            YES             NO
             ↓               ↓
       Traditional       Is there useful
       Programming            data?
                           /        \
                         NO          YES
                         ↓             ↓
                  ML may not      Can simple
                  be suitable      rules solve it?
                                  /          \
                                YES           NO
                                 ↓             ↓
                           Use simpler      Consider ML
                           solution
```

---

## **ML Terminology**

### **Dataset**

*A dataset is a collection of data used to analyze a problem or train and evaluate a machine-learning model.*

**Example:** *Suppose we want to predict house prices. Our dataset might look like*

| Area | Bedrooms | Location  |  Price |
| ---- | -------- | --------- | ------ |
| 1000 |        2 | Hyderabad |   ₹50L |
| 1500 |        3 | Hyderabad |   ₹70L |
| 2000 |        3 | Hyderabad |   ₹90L |
| 2500 |        4 | Hyderabad | ₹1.2Cr |

*The entire table is the **Dataset**.*

---

**Dataset = Collection of Examples**

*A dataset isn't necessarily a table. It depends on the type of problem.*

- **Tabular data:** *```Rows + Columns```*

     **Examples:** *```Customer | Age | Income | Churn```*

- **Image dataset**

     ```txt
     Image 1
     Image 2
     Image 3
     ...
     ```

- **Text dataset**

     ```txt
     Text 1
     Text 2
     Text 3
     ...
     ```

- **Audio dataset**

     ```txt
     Audio recording 1
     Audio recording 2
     ...
     ```

*Dataset always means Excel table. A dataset can contain many types of data.*

---

**Dataset in an ML project**

*Suppose we're building a churn prediction model. Our dataset might contain*

| Age | Tenure | Monthly Charges | Contract | Churn |
| --- | ------ | --------------- | -------- | ----- |
|  25 |      2 |             800 | Monthly  | Yes   |
|  42 |      5 |            1200 | Yearly   | No    |
|  31 |      1 |             900 | Monthly  | Yes   |

*This dataset contains examples of customers and their historical outcomes. We can use it to train a supervised ML model.*

---

**Dataset vs Database**

**Database:** *A database is a system used to store and manage data.*

**Example:**

```txt
Customer Database
       ↓
Millions of records
```

**Dataset:** *A dataset is a collection of data used for a particular analysis or ML task.*

*For example, we might extract*

```txt
Database
   ↓
Select relevant records/columns
   ↓
ML Dataset
```

---

**Dataset vs Data Point**

*Suppose*

| Age | Income | Churn |
| --- | ------ | ----- |
|  25 |  30000 | Yes   |
|  30 |  50000 | No    |
|  42 |  70000 | No    |

- *The entire table is the **Dataset***
- *One individual row is one **Observation/Instance***

```txt
Dataset
│
├── Observation 1
├── Observation 2
├── Observation 3
└── ...
```

---

**Why do we need a Dataset?**

*Because an ML model needs examples from which it can learn patterns*

```txt
Dataset
   ↓
Learning algorithm
   ↓
Model
   ↓
Predictions
```

*Without relevant data, a data-driven ML model generally has nothing useful from which to learn*

---

**Training Dataset vs Entire Dataset**

*People sometimes casually say "The dataset is used to train the model". But in a proper ML workflow, the available data may be divided into*

```txt
Available Dataset
       │
       ├── Training Data
       ├── Validation Data
       └── Test Data
```

*So the training data is only one part of the overall data used in the ML workflow*

---

### **Feature**

*A feature is an input variable or characteristic used by a machine-learning model to make a prediction*

**Example:** *Suppose we want to predict house price. Our data is*

| Area | Bedrooms | Age | Location  | Price |
| ---- | -------- | --- | --------- | ----- |
| 1000 |        2 |  10 | Hyderabad |   50L |
| 1500 |        3 |   5 | Hyderabad |   70L |
| 2000 |        3 |   3 | Hyderabad |   90L |

*Here*

```txt
Area
Bedrooms
Age
Location
```

*are **features***

*Because they are the information we give to the model to help predict ```Price```*

---

**Feature vs Target**

*Suppose*

| Age | Income | Tenure | Churn |
| --- | ------ | ------ | ----- |
|  25 |    30K |      2 | Yes   |
|  40 |    60K |      5 | No    |
|  32 |    45K |      1 | Yes   |

*If our goal is to predict Churn*

```txt
Features:
Age
Income
Tenure

Target:
Churn
```

```txt
FEATURES
   ↓
   Model
   ↓
 TARGET
```

*Features are Inputs and the Target is what we're trying to predict*

---

**Features can be different types**

*A feature doesn't have to be a number*

- **Numerical feature**

     ```txt
     Age = 25
     Income = ₹50,000
     Tenure = 3 years
     ```

- **Categorical feature**

     ```txt
     Contract = Monthly
     City = Hyderabad
     Gender = Female
     ```

- **Boolean feature**

     ```txt
     Has_Premium = True
     ```

- **Text**

     ```txt
     Review = "The service was excellent"
     ```

- **Image:** *An image itself can be input data/features depending on the ML setup.*

---

**Feature in a Table**

*Suppose*

| Age | Income | Experience | Purchased |
| --- | ------ | ---------- | --------- |
|  21 |  30000 |          0 | No        |
|  25 |  45000 |          2 | Yes       |
|  30 |  70000 |          5 | Yes       |

*We can represent it as*

```txt
Features (X)
        ↓
Age
Income
Experience

Target (y)
        ↓
Purchased
```

*In ML notation, you'll often see*

```txt
X = features
y = target
```

---

**What makes a Good Feature?**

*A useful feature should generally*
- *Contain information relevant to the target*
- *Be available at prediction time*
- *Be reasonable reliable*
- *Not leak the answer from the future or target itself*

---

**Feature Leakage**

*Suppose we're predicting "Will a customer churn next month?"*

*We use*

```txt
Age
Tenure
Monthly Charges
```

*These can potentially be available before the prediction*

*But suppose we use ```Cancellation Date```*

*If the cancellation date is recorded after the customer has already decided/started the churn process, it can reveal the answer. That's a form of data leakage*

---

**Feature Engineering**

*Sometimes raw features aren't the most useful representation*

*For example ```Date of Birth``` might be transformed into ```Age``` or ```Purchase Date``` could be transformed into ```Days Since Last Purchase```*

*Creating or transforming useful input variables is called **Feature Engineering***

---

**Feature Selection**

*Choosing which available features should be used by the model*

*Suppose you have ```100 features``` but only 20 are useful*

*You may select a subset*

```txt
100 Features
     ↓
Feature Selection
     ↓
20 Useful Features
     ↓
Model
```

*This can help simplify the model and sometimes improve performance*

---

### **Target / Label**

**What is a Target?**

*The target is the output that a machine-learning model is trying to predict.*

**Examples:** *Customer Churn*

*Suppose we have*

| Age | Tenure | Monthly Charges | Support Calls | Churn |
| --- | ------ | --------------- | ------------- | ----- |
|  25 |     12 |             799 |             2 | No    |
|  42 |      3 |            1299 |             8 | Yes   |
|  31 |     24 |             599 |             1 | No    |

*Here*
- **Features:** *Age, Tenure, Monthly Charges, Support Calls*
- **Target:** *Churn*

*The model learns from existing examples and tries to predict "Will this customer churn?"*

---

**Target in Classification**

*When the target is a category, it is a classification problem*

**Examples:**

| Problem              | Target               |
| -------------------- | -------------------- |
| Spam detection       | Spam / Not Spam      |
| Customer churn       | Yes / No             |
| Loan approval        | Approved / Rejected  |
| Disease prediction   | Disease / No Disease |
| Image classification | Cat / Dog            |

---

**Target in Regression**

*When the target is a numerical value, it is usually a regression problem*

**Examples:** *House price prediction*

| Area | Bedrooms | Location  | Price |
| ---- | -------- | --------- | ----- |
| 1200 |        2 | Hyderabad |  65 L |
| 1800 |        3 | Hyderabad |  95 L |
| 2500 |        4 | Hyderabad | 140 L |

---

**Target vs Feature**

*Suppose*

```txt
Age
Salary
Experience
Education
Job_Change
```

*If our problem is "Predict whether a person will change jobs" then*

```txt
Features → Age, Salary, Experience, Education
Target   → Job_Change
```

*But if our problem changes to "Predict the person's salary" then*

```txt
Features → Age, Experience, Education, Job_Change
Target   → Salary
```

*The target depends on the ML problem we are trying to solve. The same dataset can have different targets for different ML tasks.*

---

**X and y**

```python
X = features
y = target
```

*For example*

```python
X = df[["Age", "Income", "Tenure"]]
y = df["Churn"]
```

*So*
- *```X``` → input features*
- *```y``` → output/target*

---

**Is the Target available during prediction?**

**During Training:** *The model has ```Features + Target```*

**Example:**

```txt
Age = 25
Income = 40,000
Tenure = 12
Churn = No
```

*The model uses these known examples to learn*

**During Prediction:** *The target is unknown*

*For a new customer*

```txt
Age = 29
Income = 45,000
Tenure = 10
Churn = ?
```

*The model predicts ```Churn = No```*

*The target is known in the training data but is what we want to predict for new data.*

---

**What is a Label?**

*Label is another term commonly used for the known output in supervised learning.*

*For example*

```txt
Email → Features
Spam → Label
```

*So*

```txt
Feature → information used to make prediction
Label   → known answer
```

---

**Can the target also be given as a feature?**

*No, not when training a normal predictive model*

*Suppose*

```txt
Age
Income
Tenure
Churn
```

*and you are predicting ```Churn```. You should not do*

```txt
X = Age, Income, Tenure, Churn
y = Churn
```

*because the model is being given the answer it is supposed to predict. This can create **data leakage** and produce misleadingly good performance.*

---

### **Observation / Instance**

**What is an Observation?**

*An observation is one individual example or record in a dataset. In a tabular dataset, an observation is usually on row.*

**Example:** *Suppose we have customer data*

| Age | Income | Tenure | Churn |
| --- | ------ | ------ | ----- |
|  25 |  40000 |     12 | No    |
|  42 |  65000 |      3 | Yes   |
|  31 |  50000 |     24 | No    |

*Here*
- **Columns** → *variables/features/target*
- **Rows** → *observations*

*So we have 3 observations*

*For example, this row ```42 | 65000 | 3 | Yes``` is one observation.*

---

**Observation vs Feature**

| Age | Income | Tenure | Churn |
| --- | ------ | ------ | ----- |
|  25 |  40000 |     12 | No    |
|  42 |  65000 |      3 | Yes   |

**Observation:** *One row ```25 | 40000 | 12 | No```*

**Feature:** *One input column*

```txt
Age
Income
Tenure
```

---

**Observation does not always means a Person**

*An observation can represent different things depending on the problem*

- **Customer churn:** *```One observation = one customer```*
- **House-price prediction:** *```One observation = one house```*
- **Fraud detection:** *```One observation = one transaction```*
- **Medical prediction:** *```One observation = one patient/case```*
- **Sales forecasting:** *```One observation = one time period```*
- **Image classification:** *```One observation = one image```*

*An observation is one individual example relevant to the ML problem*

---

```txt
Dataset
   ↓
Observations / Instances
   ↓
Features + Target
```

**Example:**

```txt
Customer Dataset
       ↓
   Customer 1
   Customer 2
   Customer 3
       ↓
Age, Income, Tenure → Features
Churn                → Target
```

---

### **Training / Validation / Test Data**

*When building an ML model, we usually don't use all the available data for training. We divide the dataset into different parts so we can train the model and honestly evaluate how it performs on unseen data.*

*The three main parts are*

```txt
Dataset
   │
   ├── Training Data
   ├── Validation Data
   └── Test Data
```

---

#### **Training Data**

- *Training data is the data used to teach the model*
- *The model looks at ```Features → Target``` and learns patterns/parameters from them*

**Example:**

```txt
Age = 25
Income = 40,000
Tenure = 12
Churn = No
```

*The model uses many such examples to learn how the input features relate to the target.*

*Training data is the portion of the dataset used to train the model and learn its parameters.*

---

#### **Validation Data**

*Validation data is used to evaluate and improve model choices during development.*

*For example, suppose you're comparing*

```txt
Model A → Logistic Regression
Model B → Decision Tree
Model C → Random Forest
```

*You train them using the training data and use validation data to help decide which setup works better.*

*Validation data can also be used for*
- *Hyperparameter tuning*
- *Model selection*
- *Comparing different approaches*

*Validation data is used during model development to compare models, tune hyperparameters, and make decisions without using the final test set.*

---

#### **Test Data**

*Test data is used for the final evaluation of the selected model.*

*Suppose after development you selected Random Forest. You then evaluate the final model on the test set ```Test Data → Final Performance``` This gives you an estimate of how the model may perform on unseen real-world data.*

*Test data is a held-out dataset used for the final evaluation of a trained and selected model on unseen data.*

---

**Why do we need Three sets?**

*Imagine you train a model and evaluate it on exactly the same data. The model has already seen those examples. So getting ```Training accuracy = 98%``` doesn't necessarily mean the model will perform well on new customers.*

*That's why we need unseen data*

```txt
Training Data
      ↓
Train Model
      ↓
Validation Data
      ↓
Choose / Tune Model
      ↓
Final Model
      ↓
Test Data
      ↓
Final Evaluation
```

---

**Simple Real-World Example**

*Suppose you have 10,000 customer records. You could divide them approximately as*

```txt
Training   → 7,000
Validation → 1,500
Test       → 1,500
```

- **Training:** *The model learns from the 7000 records*
- **Validation:** *You use 1500 records to make development decisions*
- **Test:** *You use the final 1500 records only for final evaluation*

*The exact percentages are not fixed rules. The appropriate split depends on the dataset and problem.*

---

**What if we don't have enough data?**

*Sometimes the dataset is small. Instead of keeping a separate validation set, we can use cross-validation on the training data.*

*For example*

```txt
Training Data
      ↓
Cross-Validation
      ↓
Model / Hyperparameter Selection
      ↓
Test Data
      ↓
Final Evaluation
```

---

**Don't confuse Training and Test data**

**Training data:** *The model learns from these examples*

```txt
House A → ₹50L
House B → ₹70L
House C → ₹90L
```

**Test data:** *The model doesn't get the actual price while making its prediction*

```txt
House X → actual price ₹80L
```

*Then we compare*

```txt
Actual     = ₹80L
Predicted  = ₹78L
```

*to calculate the appropriate evaluation metric*

---

### **Model**

*A model is the learned mathematical/computational representation that uses input features to produce predictions or decisions.*

*OR*

*The model is what we get after an ML algorithm learns patterns from training data*

---

**Example:** *Suppose we want to predict customer churn*

*Our features are*

```txt
Age
Tenure
Monthly Charges
Support Calls
```

*Target*

```txt
Churn
```

*We give training data to an ML algorithm*

```txt
Training Data
     ↓
ML Algorithm
     ↓
Learned Model
     ↓
New Customer Data
     ↓
Prediction
```

*The resulting model can take a new customer's information and predict ```Churn = Yes```*

---

**Algorithm vs Model**

**Algorithm:** *The algorithm is the method/procedure used to learn from data.*

**Examples:**
- *Linear Regression*
- *Logistic Regression*
- *Decision Tree*
- *Random Forest*
- *KNN*
- *SVM*

**Model:** *The model is the learned result after applying the algorithm to training data.*

```txt
Algorithm + Training Data
          ↓
       Learning
          ↓
      Trained Model
```

---

**Example with Linear Regression**

- *Suppose we want to predict salary from years of experience. A linear regression model might learn a relationship such as ```Salary = 3 × Experience + 5``` The numbers in the model are learned from the training data*
- *If ```Experience = 5``` the model can produce a prediction. The exact values are not manually written by us, they are learned during training*

---

**What does a Model actually Learn?**

- *A model learns parameters from the training data.* 
- *For example, in a simple linear model ```y = wx + b``` the model learns*
     ```txt
     w → weight/coefficient
     b → intercept/bias
     ```
*These are parameters*

---

**Model during Prediction**

*Once training is complete*

```txt
Trained Model
      ↓
New Input
      ↓
Prediction
```

**Example:**

```txt
Age = 30
Tenure = 8
Monthly Charges = ₹999
Support Calls = 5
```

*The trained churn model might output ```Predicted Churn = Yes```*

*The process of using a trained model to make predictions on new data is called **Inference***

---

**Model is not the same as Dataset**

**Dataset:** *Contains examples*

```txt
Customer 1
Customer 2
Customer 3
...
```

**Model:** *Contains the learned representation/patterns used to make predictions*

```txt
Dataset
   ↓
Training
   ↓
Model
```

---

**Model is not the same as Algorithm**

*Suppose you use Decision Tree*

```txt
Decision Tree algorithm
        ↓
Training data
        ↓
Trained Decision Tree model
```

---

### **Parameters**

*A parameter is a value that the ML model learns from the training data during training.*

**Example:** *Suppose we want to predict salary based on years of experience*

*A simple model might looks like ```Salary = w × Experience + b```*

*Here*
- *```w``` = coefficient/weight*
- *```b``` = intercept/bias*

*During training the model learns suitable values for ```w``` and ```b```*

*For example it might learn*

```txt
w = 4
b = 5
```

*So the learned model becomes ```Salary = 4 × Experience + 5``` Those learned values are parameters*

---

**Why are Parameters important?**

*Parameters determine how the model behaves*

*Suppose the model learns ```w = 4``` It means the model has learned a particular relationship between experience and salary*

*If training data changes, the model may learn different parameter values*

```txt
Training Data A
      ↓
Parameters A

Training Data B
      ↓
Parameters B
```

*So parameters are learned from data, not normally manually chosen by us*

---

**Parameters vs Features**

**Feature:** *A feature is an input variable*

**Example:**

```txt
Experience
Age
Education
```

**Parameter:** *A parameter is a value learned by the model*

**Example:**

```txt
w = 4
b = 5
```

*So*

```txt
Feature → input to the model
Parameter → learned value inside the model
```

---

**Parameters vs Hyperparameters**

**Parameter:** *Learned by the model from training data*

**Examples:**
- *Linear regression coefficients*
- *Logistic regression coefficients*
- *Neural-network weights and biases*

**Hyperparameter:** *Set by us before/during training, the model does not learn it directly from the training data*

**Examples:**
- *Decision tree maximum depth*
- *K in KNN*
- *Learning rate*
- *Number of trees in Random Forest*

*Parameters are learned. Hyperparameters are chosen/tuned*

---

**Simple Analogy**

*Think about studying for an interview*

**Parameters:** *You study questions and gradually learn the answers ```Practice → Learning → Knowledge``` The knowledge you learn is similar to parameters learned from data*

**Hyperparameters:** *Before studying you decide*

```txt
Study 2 hours
Practice 50 questions
Take 3 mock interviews
```

*These are choices you make rather than something automatically learned from the practice data*

---

**Example: Logistic Regression**

*Suppose we're predicting whether a customer will churn*

*Features*

```txt
Age
Tenure
Monthly Charges
Support Calls
```

*Logistic regression learns a coefficient for each feature*

```txt
Age               → coefficient
Tenure            → coefficient
Monthly Charges   → coefficient
Support Calls     → coefficient
```

*These coefficients are parameters. The model uses them to calculate the prediction*

---

**Example: Neural Network**

*A neural network can have a huge number of parameters*

*Its parametes include ```Weights Biases```*

*During training these values are adjusted so that the model's predictions become better according to its training objective. You don't manually specify every weight. The training process learns them.*

---

### **Hyperparameters**

*A hyperparameter is a setting that we choose before or during model training that controls how the learning process or model is configured.*

**Example:** *Suppose we're using a Decision Tree*

*We can control ```Maximum depth = 5``` The tree doesn't learn ```5``` as a model parameter from the training data. We choose it. So ```max_depth → Hyperparameter``` The tree then learns its actual splitting decisions from the training data.*

---

**Parameter vs Hyperparameter**

|                    | Parameter              | Hyperparameter         |
| ------------------ | ---------------------- | ---------------------- |
| Who determines it? | Model learns it        | We choose/tune it      |
| When?              | During training        | Before/during training |
| Example            | Regression coefficient | Learning rate          |
| Example            | Neural-network weight  | Number of trees        |
| Example            | Neural-network bias    | Tree maximum depth     |

---

**Common Hyperparameter Examples**

**Decision Tree**

```txt
max_depth
min_samples_split
min_samples_leaf
```

*These control how the tree is built*

**Random Forest**

```txt
n_estimators
max_depth
```

*```n_estimators``` controls the number of trees*

**KNN**

*```k``` determines how many neighboring points are considered*

**SVM**

```txt
C
kernel
gamma
```

**Neural Networks**

```txt
learning_rate
batch_size
number_of_epochs
number_of_layers
```

---

**Why do we Tune Hyperparameters?**

*Different hyperparameter settings can produce different model performance*

*For example*

```txt
Decision Tree

max_depth = 2 → Model A
max_depth = 5 → Model B
max_depth = 15 → Model C
```

*We evaluate these choices using appropriate validation procedures and select a suitable configuration. This process is called **Hyperparameter tuning***

---

**Example: KNN**

*Suppose you're using KNN for classification. You choose ```k = 3``` The model considers the 3 nearest neighbors when making a prediction*

*If you instead choose ```k = 10``` the behavior changes. ```k``` is a hyperparameter*

*The KNN algorithm doesn't learn ```k``` as a parameter in the same way that linear regression learns coefficients*

---

**Example: Neural Network**

*Suppose we train a neural network*

*We might choose*

```txt
Learning rate = 0.001
Batch size = 32
Epochs = 20
```

*These are hyperparameters*

*During training the network learns*

```txt
Weights
Biases
```

*These are parameters*

*So*

```txt
Hyperparameters
      ↓
Control training

Parameters
      ↓
Learned during training
```

---

### **Prediction**

*A prediction is the output produced by a trained machine-learning model when it receives input data.*

---

**Basic Flow**

*After the model has been trained*

```txt
New Input
   ↓
Trained Model
   ↓
Prediction
```

**Example:**

```txt
Age = 30
Tenure = 8 months
Monthly Charges = ₹999
Support Calls = 5
```

*The trained churn model might predict ```Churn = Yes``` That output is the prediction*

---

**Prediction in Classification**

*In classification, the prediction is a category/class*

**Example:**

**Spam detection**

```txt
Email
  ↓
Model
  ↓
Spam
```

*Possible predictions ```Spam``` ```Not Spam```*

**Customer Churn**

```txt
Customer information
       ↓
      Model
       ↓
    Churn = Yes
```

*The prediction is a class label*

---

**Prediction in Regression**

*In regression, the prediction is usually a numerical value*

**Example:**

**House Price Prediction**

- **Input:**

     ```txt
     Area = 1500 sq.ft
     Bedrooms = 3
     Location = Hyderabad
     ```

- **Model:** *```↓````*
- **Prediction:** *```₹85,00,000```*

*Here ```₹85,00,000``` is the model's prediction*

---

**Prediction vs Target**

**Target:** *The actual value we want to predict*

**Prediction:** *The value produced by the model*

**Example:**

```txt
Actual Churn   = Yes
Model predicts = No
```

*So*

```txt
Target / Actual value → Yes
Prediction            → No
```

---

**Prediction vs Actual Value**

*Suppose we're predicting house prices*

```txt
Actual price      = ₹80 lakh
Predicted price   = ₹76 lakh
```

*The model's prediction is ```₹76 lakh```*

*The actual target is ```₹80 lakh```*

*We can then calculate an appropriate evaluation metric to understand how well the model performed*

---

**Classification Prediction can be more than just a Label**

*A classification model can sometimes produce a probability for each class*

*For example*

```txt
Customer
   ↓
Model
   ↓
Probability of churn = 0.82
```

*That means the model estimates an 82% probability for the churn class under that model's probability interpretation.*

*A final class prediction may then be obtained using a decision threshold*

*For example*

```txt
Probability ≥ threshold → Churn
Probability < threshold → Not Churn
```

---

**Prediction on New Data**

*Suppose the model was trained on ```10,000 customers```*

*Now a new customer arrives*

```txt
Age = 27
Income = ₹45,000
Tenure = 6
Monthly Charges = ₹899
```

*We provide these features to the trained model*

```txt
New Customer
     ↓
Trained Model
     ↓
Prediction
```

*For example ```Predicted Churn = Yes```*

---

**Prediction vs Training**

**Training:** *The model learns parameters from training data*

```txt
Training Data
     ↓
Training
     ↓
Learned Model
```

**Prediction:** *The trained model is used to produce an output*

```txt
New Data
   ↓
Trained Model
   ↓
Prediction
```

---

### **Inference**

*Inference is the process of using a trained ML model to make predictions on new, unseen data.*

---

**Basic Flow**

```txt
Training Data
      ↓
   Training
      ↓
 Trained Model
      ↓
   New Data
      ↓
   Inference
      ↓
   Prediction
```

*For example, we train a customer-churn model using historical customer data*

*Later a new customer arrives*

```txt
New Customer
     ↓
Trained Churn Model
     ↓
Inference
     ↓
Churn = Yes
```

*The process of applying the trained model to that new customer is inference*

---

**Training vs Inference**

| Training                         | Inference                       |
| -------------------------------- | ------------------------------- |
| Model learns from data           | Model is used on new data       |
| Learns parameters                | Uses learned parameters         |
| Usually computationally heavier  | Usually needs to be faster      |
| Happens during model development | Happens when making predictions |

*Training teaches the model. Inference uses the model*

---

**Real-World Example - Fraud Detection**

*Imagine a bank trains a fraud detection model using millions of historical transactions*

**Training**

```txt
Historical Transactions
          ↓
       Training
          ↓
    Fraud Detection Model
```

*Later you make a new transaction*

```txt
₹25,000 transaction
     ↓
Trained Model
     ↓
Inference
     ↓
Fraud probability / prediction
```

*The bank is performing inference when it applies the trained model to your transaction*

---

**Inference is not Retraining**

*Suppose a model is already trained. When a new customer arrives*

```txt
New Customer
     ↓
Model
     ↓
Prediction
```

*The model is not normally retrained from scratch for every customer. It is simply being used to make a prediction. That's inference*

---

**Batch Inference vs Real-Time Inference**

- **Real-Time/Online Inference:** *The model makes a prediction when a request arrives*

     **Example:**

     ```txt
     User uploads image
          ↓
     Model
          ↓
     Prediction immediately
     ```

     **Examples:**
     - *Fraud detection during a transaction*
     - *Spam detection when an email arrives*
     - *Recommendation when a user opens an app*

- **Batch Inference:** *The model makes predictions for many records together, usually on a schedule*

**Example:**

```txt
10 million customers
       ↓
Trained Model
       ↓
Predictions
       ↓
Daily customer-risk report
```

*For example a company might run predictions every night for all customers*

```txt
Real-time inference → prediction when needed
Batch inference → predictions for many records together
```

---

**ML Application Example**

*Suppose you built your ChurnGuard AI project*

*Your workflow could be*

```txt

Historical Customer Data
          ↓
       Training
          ↓
   Trained Model
          ↓
      Flask App
          ↓
New Customer Details
          ↓
       Inference
          ↓
 Churn Prediction
```

*For example*

```txt
Input:
Tenure = 3 months
Monthly Charges = ₹1299
Support Calls = 8

       ↓

Inference

       ↓

Prediction:
High risk of churn
```

*The Flask Application is using the trained model to perform inference*

---

### **Training**

*Training is the process through which a machine-learning model learns its parameters from training data*

---

**Basic Training Process**

*Suppose we want to predict customer churn*

*We have*

```txt
Features → Age, Tenure, Monthly Charges, Support Calls
Target   → Churn
```

*We provide training examples*

```txt
Training Data
      ↓
ML Algorithm
      ↓
Model learns patterns
      ↓
Parameters are adjusted
      ↓
Trained Model
```

*During training, the model's parameters are learned/adjusted using the training data*

---

**What happens during Training?**

*At a high level*

```txt
1. Give input data to model
          ↓
2. Model produces an output
          ↓
3. Compare output with actual target
          ↓
4. Calculate how wrong the model is
          ↓
5. Adjust parameters
          ↓
6. Repeat
```

*This process continues until the model reaches a suitable level of performance or the chosen training procedure stops. The exact mechanism depends on the algorithm.*

---

**Simple Example**

*Suppose the actual target is ```Actual Churn = Yes``` The model initially predicts ```Prediction = No``` The model has made an error*

*During training the learning algorithm uses an appropriate objective/loss and updates the model's parameters so that it can improve its predictions*

*After many training examples and iterations, the model may learn useful relationships between ```Customer characteristics → Churn```*

---

### **Evaluation**

*Evaluation is the process of measuring how well a trained machine-learning model performs on data it should be tested against.*

---

**Why do we Evaluate a Model?**

*Training a model doesn't automatically mean it is good*

*Suppose you train a churn model*

*It predicts*

```txt
Customer 1 → No
Customer 2 → Yes
Customer 3 → No
...
```

*We need to know "How accurate or useful are these predictions?"*

---

**Basic Evaluation Flow**

```txt
Trained Model
      ↓
Validation/Test Data
      ↓
Predictions
      ↓
Compare with Actual Target
      ↓
Evaluation Metric
      ↓
Model Performance
```

*For example*

```txt
Actual      → Yes
Prediction  → Yes
```

*The model got that example correct*

*Another*

```txt
Actual      → Yes
Prediction  → No
```

*The model got that example wrong. The evaluation process summarizes these kinds of outcomes*

---

**Evaluation depends on the Problem**

*You don't use the same metric for every ML problem*

**Classification:** *Common metric include*
- *Accuracy*
- *Precision*
- *Recall*
- *F1-score*
- *ROC-AUC*

**Regression:** *Common metrics include*
- *MAE*
- *MSE*
- *RMSE*
- *R²*

*The appropriate evaluation metric depends on the ML problem and business objective*

---

**Evaluation vs Training**

**Training:** *The model learns its parameters*

```txt
Training Data
      ↓
Learn Parameters
```

**Evaluation:** *We measure how well the trained model performs*

```txt
Trained Model
      ↓
Validation/Test Data
      ↓
Measure Performance
```

---

**Validation vs Test Evaluation**

```txt
Training → Learn

Validation → Development decisions

Test → Final evaluation
```

**Validation evaluation:** *Used while developing the model*

*For example*

```txt
Model A → Validation score = 82%
Model B → Validation score = 87%
```

*You may use this information to make development decisions*

**Test evaluation:** *After selecting the final model, evaluate it on the held-out test set*

```txt
Final Model
     ↓
Test Data
     ↓
Final Performance
```

---

**Simple Example**

*Suppose you have 100 test examples. The model correctly predicts 90 of them.*

*For a simple accuracy calculation*

```txt
Accuracy = Correct Predictions / Total Predictions

         = 90 / 100

         = 90%
```

*So the model's accuracy is 90%. But don't conclude that 90% accuracy automatically means the model is good. For some problems accuracy can be misleading.*

---

**Evaluation is more than One number**

*Suppose two classification models have*

```txt
Model A → Accuracy = 95%
Model B → Accuracy = 92%
```

*You shouldn't automatically choose ```Model A``` just because its accuracy is higher. The right metric depends on the cost of different errors.*

*For example, in some fraud-detection problems, missing a fraudulent transaction may be much more important than incorrectly flagging a legitimate transaction.*

*So you may care more about recall, precision or another appropriate metric*

*This is why **Metric selection should be connected to the actual problem**.*

---

### **Loss**

