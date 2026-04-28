# Task 5: Mental Health Support Chatbot (Fine-Tuned)

## Objective
Build a basic chatbot that provides supportive and empathetic responses for stress, anxiety, and emotional wellness.

## Project Overview
This project involves fine-tuning a small LLM (**DistilGPT2**) using the **EmpatheticDialogues** dataset from Facebook AI. The goal is to create a chatbot that doesn't just answer questions but responds with empathy and emotional intelligence.

## Dataset
- **Name:** EmpatheticDialogues (Facebook AI)
- **Description:** A dataset of 25k conversations grounded in specific emotional contexts.
- **Usage:** Used to train the model to recognize emotional cues and respond appropriately.

## Model
- **Base Model:** `distilgpt2`
- **Architecture:** Transformer-based Causal Language Model.
- **Fine-tuning:** Performed using Hugging Face's `Trainer` API.

## Key Features
- **Empathetic Tone:** Specifically trained to be gentle and supportive.
- **Streamlit Interface:** A clean, user-friendly web interface for real-time interaction.
- **Modular Code:** Separated training logic (`train.py`) and application logic (`app.py`).

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-tune the Model (Optional)
To train the model yourself, run:
```bash
python train.py
```
*Note: This may take several hours depending on your hardware.*

### 3. Run the Chatbot
```bash
streamlit run app.py
```

## Repository Structure
- `mental_health_chatbot.ipynb`: Comprehensive notebook with data exploration, preprocessing, and training explanation.
- `train.py`: Standalone script for fine-tuning the model.
- `app.py`: Streamlit application for the chatbot interface.
- `requirements.txt`: List of required Python packages.

## Results and Findings
- **Loss:** The model's cross-entropy loss decreased significantly during the first 3 epochs.
- **Empathetic Quality:** The model shows a noticeable improvement in using supportive language compared to the base `distilgpt2` model.
- **Limitations:** As a small model, it may occasionally produce repetitive or generic responses. Further tuning with a larger model like `GPT-Neo` or `Mistral-7B` would improve depth.
