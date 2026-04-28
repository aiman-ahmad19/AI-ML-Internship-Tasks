import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import random

# Page configuration
st.set_page_config(
    page_title="MindfulBot | Your Emotional Sanctuary",
    page_icon="🌈",
    layout="wide"
)

# Custom CSS for a more attractive and colorful UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testimonial="user"] {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
    }
    .stChatMessage[data-testimonial="assistant"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .support-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2575fc;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="main-header">
        <h1>🌈 MindfulBot</h1>
        <p>Your empathetic companion for a calmer mind and heart.</p>
    </div>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "./mental_health_model"
    if not os.path.exists(model_path):
        model_name = "distilgpt2"
    else:
        model_name = model_path
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return tokenizer, model, device

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Beautiful & Functional
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/mental-health-awareness-concept_23-2148531011.jpg", use_container_width=True)
    st.title("🌱 Wellness Corner")
    st.markdown("""
    ### Daily Affirmation
    *"You are capable of handling whatever this day throws at you."*
    
    ---
    ### How I can help:
    - 🧘 **Stress Management**
    - 💭 **Emotional Support**
    - 👂 **Active Listening**
    - 🌈 **Positive Encouragement**
    """)
    
    if st.button("✨ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.info("💡 **Tip:** Try telling me how your day went or what's currently on your mind.")

# Load model
tokenizer, model, device = load_model()

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("How are you feeling today?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Knowledge base for high-quality responses
        support_responses = {
            "greetings": [
                "Hello! I'm so glad you're here. How can I support you today? 🌸",
                "Hi there! I'm here to listen. How are you feeling in this moment? ✨",
                "Hey! It's good to connect with you. What's on your heart today? 🌈"
            ],
            "sad": [
                "I'm so sorry you're feeling sad. It's okay to not be okay. Do you want to talk more about what's making you feel this way? 💙",
                "I hear you, and I'm here for you. Sending you a virtual hug. What do you think might help you feel a little more comforted right now? 🫂",
                "It sounds like things are heavy for you right now. I'm listening. Take all the time you need. 🕯️"
            ],
            "stress": [
                "Stress can be so overwhelming. Let's take a deep breath together... In for 4, hold for 4, out for 4. What's the biggest thing on your plate right now? 🌿",
                "I can tell you're carrying a lot. Remember, you don't have to handle everything at once. What's one small thing we can focus on? 🍃",
                "You're doing your best, and that is enough. Stress is tough, but you are tougher. How can I best support you through this? 🌤️"
            ],
            "anxious": [
                "I hear the anxiety in your words. You are safe here. Let's try to ground ourselves. Can you tell me three things you can see around you right now? ⚓",
                "Anxiety is like a storm, but it will pass. I'm right here with you. What's the main thought that's making you feel uneasy? 🌊",
                "It's okay to feel anxious. Your feelings are valid. Let's take it one step at a time. I'm listening. 🕊️"
            ],
            "lonely": [
                "I'm here with you. Even though I'm an AI, I care about your well-being. What's making you feel lonely today? 🤝",
                "You're not alone in feeling this way. I'm here to keep you company. Want to tell me about your day? 🌟"
            ],
            "tired": [
                "It sounds like you've been working really hard. Please remember to be kind to yourself. Have you had a chance to rest today? 😴",
                "Exhaustion can make everything feel harder. I'm here if you just need to vent. What's been draining your energy lately? 🔋"
            ],
            "angry": [
                "It's completely valid to feel angry. It's often a sign that something needs to change. What happened to trigger this feeling? ⚡",
                "I'm here to listen while you let it out. Anger can be a lot to carry. What's on your mind? 🌪️"
            ],
            "job": [
                "I hear how much this job search is weighing on you. The uncertainty of the future can be really scary. What's been the hardest part of the process so far? 💼",
                "It's completely understandable to feel worried about your career. Your worth isn't defined by your job status. I'm here to listen if you want to vent about the search. 🌟",
                "Job hunting is an emotional rollercoaster. Please try to be kind to yourself during this time. What's one small thing you can do for yourself today to de-stress? 🌤️"
            ]
        }

        prompt_lower = prompt.lower().strip()
        
        with st.chat_message("assistant"):
            with st.spinner("Reflecting..."):
                response = ""
                # Priority 1: Smart Keyword Matching (Expanded)
                if any(word in prompt_lower for word in ["hi", "hello", "hey", "hy", "good morning", "good evening"]):
                    response = random.choice(support_responses["greetings"])
                elif any(word in prompt_lower for word in ["sad", "depressed", "unhappy", "cry", "miserable"]):
                    response = random.choice(support_responses["sad"])
                elif any(word in prompt_lower for word in ["job", "career", "work", "interview", "hired", "employment"]):
                    response = random.choice(support_responses["job"])
                elif any(word in prompt_lower for word in ["stress", "busy", "overwhelmed", "exam", "pressure"]):
                    response = random.choice(support_responses["stress"])
                elif any(word in prompt_lower for word in ["anxious", "scared", "worry", "worried", "panic", "fear", "nervous"]):
                    response = random.choice(support_responses["anxious"])
                elif any(word in prompt_lower for word in ["lonely", "alone", "nobody", "isolated"]):
                    response = random.choice(support_responses["lonely"])
                elif any(word in prompt_lower for word in ["tired", "exhausted", "sleepy", "burnt out"]):
                    response = random.choice(support_responses["tired"])
                elif any(word in prompt_lower for word in ["angry", "mad", "furious", "annoyed", "hate"]):
                    response = random.choice(support_responses["angry"])
                
                # Priority 2: AI Generation for complex queries
                if not response:
                    # More structured prompt specifically for DistilGPT2 to keep it on track
                    structured_prompt = (
                        "The following is a conversation with a kind and empathetic mental health assistant.\n"
                        "User: I am feeling very down today.\n"
                        "Assistant: I'm so sorry to hear that. I'm here to listen. What's on your mind?\n"
                        f"User: {prompt}\n"
                        "Assistant:"
                    )
                    
                    inputs = tokenizer(structured_prompt, return_tensors="pt").to(device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=40, # Keep it short to prevent rambling
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        top_k=30,
                        top_p=0.85,
                        temperature=0.5, # Lower temperature for even more stability
                        repetition_penalty=1.3,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract only the last Assistant response
                    if "Assistant:" in full_response:
                        parts = full_response.split("Assistant:")
                        response = parts[-1].strip()
                    else:
                        response = full_response.replace(structured_prompt, "").strip()
                    
                    # Intensive cleaning to remove hallucinations or fragments
                    response = response.split("User:")[0].split("Assistant:")[0].strip()
                    response = response.split("\n")[0].strip() # Take only the first line

                # Final quality check: if it looks like gibberish or is too short
                if not response or len(response) < 10 or "_" in response[:10] or "tournament" in response.lower() or "..." in response[:5]:
                    response = "I hear you, and I want you to know that your feelings are completely valid. It sounds like you're going through a challenging time. I'm here to listen—would you like to tell me more about what's on your mind? ❤️"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.markdown("### 🧘 Quick Relief")
    st.markdown("""
    <div class="support-card">
        <h4>4-7-8 Breathing</h4>
        <p>1. Inhale for <b>4</b> seconds</p>
        <p>2. Hold for <b>7</b> seconds</p>
        <p>3. Exhale for <b>8</b> seconds</p>
    </div>
    <div class="support-card">
        <h4>Grounding Exercise</h4>
        <p>Find 5 things you can see, 4 things you can touch, and 3 things you can hear.</p>
    </div>
    <div class="support-card" style="border-left: 5px solid #6a11cb;">
        <h4>Self-Care Reminder</h4>
        <p>• Drink a glass of water<br>
        • Step outside for fresh air<br>
        • Be kind to your inner voice</p>
    </div>
    """, unsafe_allow_html=True)
