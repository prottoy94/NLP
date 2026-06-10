from langchain_core.prompts import PromptTemplate  # Fixed import

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
   
3. **Important Rules:**  # Removed the extra 'a'
   - If specific information is not available in the paper, respond with exactly: "Insufficient information available"
   - Do not guess or hallucinate information
   - Stay strictly within the paper's content

**Format your response as:**
- Clear section headers
- Bullet points for key ideas
- Code blocks for mathematical implementations
- Separators between different sections

Now provide the summary for "{paper_input}" in {style_input} style with {length_input} length.""",
    input_variables=["paper_input", "style_input", "length_input"],  # Added commas
    validate_template=False  # Added comma above
)

template.save("paper_summary_template.json")
print("Template saved successfully!")