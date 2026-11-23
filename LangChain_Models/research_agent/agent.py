import sys
import uuid

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import store
from llm import get_llm, DEFAULT_PROVIDER, PROVIDERS
from tools import build_tools, is_rate_limit


SYSTEM_PROMPT = """You are a research paper assistant.

Follow this workflow:
1. Use search_arxiv to find papers.
2. Use summarize_paper for every paper you need to cover. It loads the paper by itself.
3. Use compare_papers when the user asks to compare two or more papers.
4. Use save_notes at the end to save the summaries and the comparison table.

Uploaded papers:
- The user can upload their own PDF papers. Their ids start with "upload-", e.g. "upload-my-paper".
- Use list_uploaded_papers when the user talks about "my paper" or "the uploaded paper" without giving its id.
- Uploaded papers work with load_paper, summarize_paper, ask_paper and compare_papers, just like arXiv papers.
- search_arxiv cannot find uploaded papers, so never search for them.

Other rules:
- Use ask_paper only for a specific question about one paper.
- Always cite papers by their paper id (arXiv id or upload id).
- Never invent paper content. Only use what the tools returned.
- If a tool returns an ERROR, tell the user what went wrong instead of guessing.
- In your final answer, include the summaries, the comparison table and the saved note path.
"""


def build_agent(provider=DEFAULT_PROVIDER, memory=None):
    if memory is None:
        memory = InMemorySaver()

    llm = get_llm(provider)
    tools = build_tools(provider)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent


def print_tool_steps(messages):
    for message in messages:
        if message.type == "ai" and message.tool_calls:
            for tool_call in message.tool_calls:
                args_text = str(tool_call["args"])
                if len(args_text) > 100:
                    args_text = args_text[:100] + "..."
                print(f"  [tool] {tool_call['name']} {args_text}")


def ask_agent(agent, user_input, thread_id):
    config = {"configurable": {"thread_id": thread_id}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )
    messages = response["messages"]
    return messages


def main():
    if len(sys.argv) > 1:
        provider = sys.argv[1]
    else:
        provider = DEFAULT_PROVIDER

    if provider not in PROVIDERS:
        print(f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}")
        sys.exit(1)

    store.load_cache()
    agent = build_agent(provider)
    thread_id = str(uuid.uuid4())
    seen_messages = 0

    print(f"Research Paper Agent ({provider})")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting agent. Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            messages = ask_agent(agent, user_input, thread_id)
        except Exception as error:
            if is_rate_limit(error):
                print("Agent error: the free-tier rate limit was reached. Wait a while or switch provider.\n")
            else:
                print(f"Agent error: {str(error)[:300]}\n")
            continue

        new_messages = messages[seen_messages:]
        seen_messages = len(messages)

        print_tool_steps(new_messages)

        answer = messages[-1].text
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
