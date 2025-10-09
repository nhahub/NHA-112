# NHA-112
# 🧠 End-to-End Customer Churn Prediction Pipeline

An **end-to-end Data Engineering project** to build, deploy, and automate a **Customer Churn Prediction Service**.  
This project was developed as part of the **Digital Egypt Pioneers Initiative (DECI)**.

---

## 📘 1. About The Project

The goal of this project is to move beyond a simple data science analysis and build a **robust, automated Data Engineering pipeline**.  

The system:
- Ingests raw customer data  
- Validates and cleans it  
- Trains a machine learning model  
- Serves predictions via a **containerized REST API**

This approach emphasizes **production-readiness**, **automation**, and **reliability** — key principles of a real-world Data Engineering solution.

---

## ⚙️ 2. Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Core Libraries** | Pandas, Scikit-learn, NumPy |
| **API Framework** | Flask |
| **Containerization** | Docker |

---

## 🧩 3. Project Architecture

The system follows a **layered pipeline architecture** for modularity and maintainability.  
Each layer is designed to be developed, tested, and deployed independently.

### 🔹 Layers Overview

1. **Ingestion Layer**  
   - Acquires raw data from the source (`/data/raw`)

2. **Processing Layer**  
   - Validates schema, cleans data (handles nulls, encodes categoricals), performs feature engineering  
   - Saves output to `/data/processed`

3. **Model Training Layer**  
   - Loads processed data  
   - Trains and evaluates a churn prediction model  
   - Saves versioned model artifacts to `/models`

4. **Model Serving Layer**  
   - Flask application that loads the latest model  
   - Exposes a `/predict` REST API endpoint for real-time predictions

---

### 📊 Data Flow Diagram (DFD)

_Add your data flow diagram here — e.g., `/docs/dfd.png`._

### 🔄 API Interaction (Sequence Diagram)

_Add your API sequence diagram here — e.g., `/docs/api_sequence.png`._

---

## 🚀 4. Getting Started

Follow these steps to set up and run the project locally or inside Docker.

---

### 🧱 Prerequisites

- **Python 3.9+**
- **Docker Desktop** (or Docker Engine)

---

### 🪜 Installation & Setup (Local Environment)

```bash
# 1️⃣ Clone the repository
git clone https://github.com/your_username/your_repository_name.git
cd your_repository_name

# 2️⃣ Create a virtual environment
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
```
