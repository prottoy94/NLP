from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template= ChatPromptTemplate([
    ("system", "You are a helpful and {domain} assistant."),
    ("human", "Explain in simple terms, what is {input}?"),
])

prompt=chat_template.invoke({'domain': 'AI','input': 'ChatGPT'})
print(prompt)