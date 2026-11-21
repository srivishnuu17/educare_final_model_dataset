# 🎓 EduCareer AI
### A Domain-Specific Small Language Model for Education and Career Guidance

EduCareer AI is an AI-powered education and career guidance system designed to help students explore educational programs, career opportunities, employment trends, and occupation-related salary information through natural-language interactions.

The project combines a fine-tuned Small Language Model (SLM), Machine Learning, dataset-based information retrieval, and a conversational AI interface to make education and career information more accessible.

---

## 🚀 Project Overview

Students often struggle to find reliable and connected information about courses, qualifications, career opportunities, employment growth, and salaries.

These details are generally distributed across different sources, making career planning difficult.

**EduCareer AI aims to address this problem** by bringing education and career-related information into a single AI-powered assistant.

### Key Objectives

- Provide accessible education and career guidance.
- Connect educational qualifications with related occupations.
- Present employment trends and occupation-related wage information.
- Support natural-language question answering.
- Use machine learning to estimate occupation median wages.
- Explore domain-specific SLM fine-tuning for education and career guidance.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| AI Career Assistant | Conversational assistance for education and career questions |
| Education Guidance | Information about programs, credentials and qualifications |
| Career Exploration | Connects education programs with occupations |
| Employment Insights | Provides employment growth and job-opening information |
| Wage Prediction | Predicts occupation median wages using Random Forest |
| Dataset Retrieval | Retrieves relevant question-answer information using TF-IDF and cosine similarity |
| Ollama Integration | Runs a locally hosted conversational model |
| Fine-Tuned SLM | Fine-tuned FLAN-T5-small for domain-specific question-answering experiments |

---

## 🧠 Model Architecture

EduCareer AI combines multiple AI and machine-learning components.

### 1. Fine-Tuned Small Language Model

- Base model: `google/flan-t5-small`
- Architecture: Encoder–Decoder Transformer
- Approximate parameters: 77 million
- Fine-tuned using education and career question-answer data.
- Training epochs: 2
- Learning rate: 5e-5
- Maximum input length: 128 tokens
- Maximum target length: 96 tokens

The fine-tuned model checkpoint is maintained separately from the current chatbot's Ollama model.

### 2. Random Forest Wage Prediction

A Random Forest Regression model is used to estimate occupation median annual wages.

**Target variable:**
`occupation_median_wage_2024_usd`

**Model configuration:**

- Algorithm: RandomForestRegressor
- Number of estimators: 200
- Maximum depth: 12
- Training/testing split: 80:20
- Random state: 42

**Recorded evaluation results:**

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | $1,583.31 |
| Root Mean Squared Error (RMSE) | $8,439.60 |
| R² Score | 0.9123 |

These results are from the recorded model evaluation on the project dataset.

Note: The target represents occupation-level median annual wages, not guaranteed individual salaries.

### 3. Dataset Retrieval System

The chatbot uses:

- TF-IDF Vectorization
- Unigram and bigram features
- Cosine Similarity
- Similarity threshold: 0.12

This component retrieves relevant information from the prepared question-answer dataset.

### 4. Ollama Integration

Ollama is used to run a local conversational model.

**Base model:** `llama3.2:1b`

**Custom Ollama model name:** `educareer-ai`

A custom Modelfile provides EduCareer AI-specific system instructions for education and career guidance.

---

## 📊 Dataset Information

The original dataset contains education, qualification, occupation, employment, and wage-related information.

| Dataset Information | Quantity |
|---|---:|
| Original records | 999 |
| Original columns | 72 |
| Training QA records | 2,926 |
| Validation QA records | 375 |
| Testing QA records | 475 |
| Total QA records | 3,776 |

### Important Dataset Categories

- Education programs and credentials
- CIP titles and CIP families
- Occupation information
- Employment statistics
- Employment growth percentages
- Annual job openings
- Median annual occupation wages
- AI and software occupation indicators

### Dataset Preparation Workflow

1. Inspect the original dataset.
2. Identify relevant education and occupation fields.
3. Transform structured information into question-answer pairs.
4. Clean and organize the generated records.
5. Divide the prepared dataset into training, validation, and testing sets.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Small Language Model | FLAN-T5-small |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF, Cosine Similarity |
| Conversational Model Runtime | Ollama |
| Base Ollama Model | Llama 3.2:1B |
| Model Hub | Hugging Face |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Development Environment | Jupyter Notebook, VS Code |

---

## 📁 Project Structure

```text
EDUCAREER_AI_SUBMISSION/
│
├── dataset/
│   ├── slm_dataset.csv.csv
│   ├── train_data.csv
│   ├── validation_data.csv
│   └── test_data.csv
│
├── model/
│   ├── edu_app.py
│   ├── edu_app_backup.py
│   ├── education_wage_model.pkl
│   ├── test_wage_model.py
│   ├── requirements.txt
│   └── EduCareer_AI_Model/
│       └── Fine-tuned FLAN-T5 checkpoint files
│
├── notebook/
│   └── EduCareer_AI_Training.ipynb
│
└── ollama/
    ├── Modelfile
    ├── readme.md
    ├── ollama_commands.txt
    └── run_educareer.bat
```

---

## ⚙️ Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or higher
- Git
- Ollama
- Hugging Face account (for accessing hosted model files, where required)

### Step 1: Clone the Repository

```bash
git clone https://github.com/srivishnuu17/EduCareer-AI-SLM.git

cd EduCareer-AI-SLM
```

### Step 2: Create a Virtual Environment

```bash
python -m venv slm_env
```

Activate it on Windows:

```bash
slm_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r model/requirements.txt
```

### Step 4: Install Ollama

Download Ollama:

https://ollama.com/download

Verify installation:

```bash
ollama --version
```

### Step 5: Download the Base Model

```bash
ollama pull llama3.2:1b
```

### Step 6: Create the Custom EduCareer AI Model

Navigate to the Ollama directory:

```bash
cd ollama
```

Create the custom model:

```bash
ollama create educareer-ai -f Modelfile
```

Verify:

```bash
ollama list
```

Run:

```bash
ollama run educareer-ai
```

### Step 7: Run the Streamlit Application

Open another terminal in the project root directory.

```bash
streamlit run model/edu_app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

**Note:** The application requires the necessary dataset and model artifacts to be available at the paths expected by the source code. Check those paths before running it.

---

## 🤗 Hugging Face Resources

**Hugging Face Profile:**

https://huggingface.co/srivishnuu1707

**Fine-Tuned FLAN-T5 Model:**

https://huggingface.co/srivishnuu1707/educareer-ai-flan-t5

**Dataset Repository:**

https://huggingface.co/datasets/srivishnuu1707/edu_care_ai_final_model_dataset

The repositories contain the project's model and dataset resources.

---

## 🧪 Testing and Evaluation

The project includes testing of:

- Dataset preparation and splitting.
- Fine-tuned SLM response generation.
- Dataset-based information retrieval.
- Random Forest wage prediction.
- Ollama conversational responses.
- Streamlit chatbot functionality.

The Random Forest model has recorded evaluation metrics. The fine-tuned SLM and conversational components require further evaluation for answer relevance, factual correctness, and numerical consistency.

---

## 🔮 Future Enhancements

- Integrate the fine-tuned FLAN-T5 model directly into the chatbot.
- Improve numerical answer accuracy and reduce hallucinations.
- Add personalized career recommendations.
- Expand the education and occupation dataset.
- Improve model evaluation using appropriate NLP metrics.
- Add multilingual education and career guidance.
- Deploy the application through a cloud-based infrastructure.

---

## 🎯 Project Outcome

EduCareer AI demonstrates how structured education and employment data can be transformed into an AI-powered guidance system by combining Small Language Models, Machine Learning, Natural Language Processing, and conversational AI.

The project aims to make education and career exploration simpler, more accessible, and more data-driven.

---

## 👨‍💻 Project Team

**Institution:** JAIN (Deemed-to-be University), Bengaluru  
**Programme:** B.Tech – Computer Science and Engineering (AI & ML)

| Team Member | USN |
|---|---|
| Sree Priya | 25BTRCL128 |
| Siva Nandini | 25BTRCL136 |
| Param Rajyaguru | 25BTRCL180 |
| Sri Vishnu | 25BTRCL183 |

---

**EduCareer AI – Connecting Education, Skills and Career Opportunities through AI.**
