import streamlit as st
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Telecom AI Brand Intelligence", layout="wide")
st.title("📡 Telecom AI Brand Intelligence System")
st.write("Real-Time Sentiment Analysis, Sarcasm Detection & RAG Resolution Engine")

@st.cache_resource
def load_ai_models():
    model_path = "Devarahaasan/telecom-sarcasm-roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return tokenizer, model, embed_model

tokenizer, model, embed_model = load_ai_models()

policies = [
    "TELECOM COMPANY OFFICIAL POLICY DOCUMENT",
    "1. NETWORK ISSUES RESOLUTION:\n- For 5G buffering and poor indoor signals, technical support must reset the local tower binding within 24 hours.\n- If a customer experiences more than 10 call drops in a day, an automated credit of 50MB data is applied.",
    "2. BILLING AND DEDUCTIONS:\n- Wrong balance deductions must be refunded to the source account within 3 working days after validation.\n- Customers can raise queries for extra charges via the mobile application under 'Billing Dispute'.",
    "3. CUSTOMER SERVICE TICKET ESCALATION:\n- If a customer support agent fails to solve an issue, the ticket escalates to Level 2 support automatically.\n- Response time for normal text queries is 2 hours maximum."
]

policy_embeddings = embed_model.encode(policies)
faiss_index = faiss.IndexFlatL2(policy_embeddings.shape[1])
faiss_index.add(np.array(policy_embeddings).astype('float32'))

st.subheader("🕵️‍♂️ Analyze Customer Feedback (English / Tanglish)")
user_input = st.text_input("Enter customer comment here:", "Wow Jio, semma fast network speed! WhatsApp text message send ஆக 2 மணிநேரம் ஆகுது. Super!")

if st.button("Run AI Intelligence Diagnostics"):
    if user_input.strip() != "":
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        
        labels_map = {0: 'Positive', 1: 'Neutral', 2: 'Negative', 3: 'Sarcastic'}
        detected_sentiment = labels_map[pred_id]
        
        text_lower = user_input.lower()
        if "speed" in text_lower or "signal" in text_lower or "network" in text_lower or "buffering" in text_lower or "drop" in text_lower:
            detected_category = "Network"
        elif "charge" in text_lower or "billing" in text_lower or "money" in text_lower or "balance" in text_lower or "deduct" in text_lower or "recharge" in text_lower:
            detected_category = "Billing"
        else:
            detected_category = "General"
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Sentiment", detected_sentiment)
        with col2:
            st.metric("Identified Category", detected_category)
            
        st.write("---")
        st.subheader("🤖 Automated Smart Resolution Generator (RAG)")
        if detected_sentiment in ['Negative', 'Sarcastic']:
            query_vector = embed_model.encode([user_input]).astype('float32')
            distances, indices = faiss_index.search(query_vector, k=1)
            
            matched_idx = indices[0][0]
            if detected_category == "Billing": matched_idx = 2
            elif detected_category == "Network": matched_idx = 1
                
            st.warning(f"⚠️ Action Required! Found Matching Policy:\n\n{policies[matched_idx]}")
        else:
            st.success("✅ No immediate policy action required for positive/neutral feedback.")
    else:
        st.error("Please enter a valid telecom review string.")
