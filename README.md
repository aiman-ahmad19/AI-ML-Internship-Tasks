# 🚀 AI/ML Internship Tasks Portfolio

This repository showcases my hands-on work on core Artificial Intelligence and Machine Learning tasks, covering **data analysis, predictive modeling, and conversational AI**.

Each task demonstrates a different stage of the ML pipeline — from understanding data to building intelligent systems.

---

## Tasks Summary

| Task   | Domain           | Key Focus                            |
| ------ | ---------------- | ------------------------------------ |
| Task 1 | Data Analysis    | Data exploration & visualization     |
| Task 3 | Machine Learning | Classification & model evaluation    |
| Task 5 | NLP / AI         | Chatbot & language model fine-tuning |

---

##  Task 1: Iris Dataset Analysis & Visualization

**Goal:**
Understand data patterns using statistical analysis and visualizations.

**What I Did:**

* Loaded and inspected dataset using pandas
* Performed statistical analysis (`info()`, `describe()`)
* Created visualizations:

  * Scatter plots → feature relationships
  * Histograms → data distribution
  * Box plots → outlier detection

**Key Insight:**
Feature relationships (e.g., petal length vs width) clearly separate species, showing that the dataset is highly structured and suitable for classification.

---

## Task 3: Heart Disease Prediction

**Goal:**
Predict the likelihood of heart disease using clinical data.

**Approach:**

* Cleaned dataset (handled missing values)
* Performed EDA to understand feature impact
* Built classification models:

  * Logistic Regression
  * Decision Tree

**Evaluation Metrics:**

* Accuracy
* ROC-AUC
* Confusion Matrix

**Results:**

* Logistic Regression achieved **~87% accuracy and ~0.95 ROC-AUC**
* Outperformed Decision Tree in generalization

**Key Insight:**
Clinical features like **age, maximum heart rate (thalach), and oldpeak** are strong predictors of heart disease risk.

---

## Task 5: Mental Health Support Chatbot

**Goal:**
Develop an empathetic chatbot capable of responding to emotional and mental health concerns.

**Approach:**

* Fine-tuned a transformer-based language model (DistilGPT2 / GPT-Neo / Mistral)
* Used **EmpatheticDialogues dataset** for training
* Focused on:

  * Emotional tone
  * Supportive responses
  * Safe conversational design

**Deployment:**

* Built a simple interface (CLI / Streamlit) for interaction

**Key Insight:**
Fine-tuning on human conversational data significantly improves the model’s ability to generate **empathetic and context-aware responses**.

---

## Tech Stack

* **Python**
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **NLP / LLMs:** Hugging Face Transformers

---

## Repository Structure

```
AI-ML-Internship-Tasks/
│
├── Task1_Iris_Visualization/
├── Task3_Heart_Disease_Prediction/
├── Task5_Mental_Health_Chatbot/
│
└── README.md
```

---

## What This Repository Demonstrates

* Strong foundation in **data preprocessing and EDA**
* Ability to build and evaluate **machine learning models**
* Practical experience with **LLMs and chatbot development**
* Understanding of **real-world problem solving in AI**

---

## 👤 Author

**Aiman**
