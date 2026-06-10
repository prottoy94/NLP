from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt

import streamlit as st
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.header("Google Generative AI Chatbot")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file")
    print("Please add: GOOGLE_API_KEY=your-key-here")
    exit(1)

try:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  
        temperature=0.9,
        max_output_tokens=2048,    
        google_api_key=api_key
    )
except Exception as e:
    print(f"Error: {e}")

paper_input = st.selectbox(
    "Select Research Paper Name", 
    [
        "Attention Is All You Need", 
        "BERT: Pre-training of Deep Bidirectional Transformers", 
        "GPT-3: Language Models are Few-Shot Learners", 
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length", 
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

templete = load_prompt("paper_summary_template.json")
#Fill the placeholders in the template with user inputs
# Use the correct variable name "templete" (as you defined it)
promt = templete.invoke(input={
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("Send"):
    st.text("Generating response...")
    result=llm.invoke(promt)
    st.text("Response:")
    st.write(result.content)