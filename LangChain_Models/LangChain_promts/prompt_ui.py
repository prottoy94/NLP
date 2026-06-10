from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.header("Google Generative AI Chatbot")
user_input = st.text_input("Enter your message:", key="user_input")

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

if st.button("Send"):
    st.text("Generating response...")
    result=llm.invoke(user_input)
    st.text("Response:")
    st.write(result.content)