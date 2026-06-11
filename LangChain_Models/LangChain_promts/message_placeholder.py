from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful and customer supportive assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

chat_history = []
with open("LangChain_promts/chat_history.txt", "r") as file:
    for line in file:
        chat_history.append(line.strip())
    
prompt = chat_template.invoke({
    "input": "What is the status of my order?",
    "chat_history": chat_history
})

print(prompt)