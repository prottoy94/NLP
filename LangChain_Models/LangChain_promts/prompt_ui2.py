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

template = PromptTemplate(
    template="""You are an expert research paper summarizer. Summarize the research paper titled "{paper_input}" according to the following specifications:

**Style:** {style_input}
**Length:** {length_input}

**Your response must include:**

1. **Mathematical Details:**
   - Extract and present key mathematical equations from the paper
   - For each important equation, provide a simple code example (Python/pseudocode) that demonstrates the concept
   
2. **Analogies:**
   - Create relatable, real-world analogies to explain complex ideas
   - Make analogies intuitive and easy to understand
   
3. **Important Rules:**
   - If specific information is not available in the paper, respond with exactly: "Insufficient information available"
   - Do not guess or hallucinate information
   - Stay strictly within the paper's content

**Format your response as:**
- Clear section headers
- Bullet points for key ideas
- Code blocks for mathematical implementations
- Separators between different sections

Now provide the summary for "{paper_input}" in {style_input} style with {length_input} length.""",input_variables=["paper_input","style_input","length_input"])
#Fill the placeholders in the template with user inputs
promt = template.invoke(input={
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("Send"):
    st.text("Generating response...")
    result=llm.invoke(promt)
    st.text("Response:")
    st.write(result.content)