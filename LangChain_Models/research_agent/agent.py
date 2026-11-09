import argparse
import uuid

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from llm import DEFAULT_PROVIDER, PROVIDERS, get_llm
from tools import build_tools

SYSTEM_PROMPT = (
    "You are a research paper assistant. "
    "Follow this workflow: search_arxiv -> load_paper -> summarize_paper -> "
    "compare_papers -> save_notes. "
    "Use search_arxiv to find papers. "
    "Use summarize_paper for every paper you were asked to cover (it loads the paper by itself). "
    "Use ask_paper only for specific questions about one paper. "
    "Use compare_papers when the user asks to compare two or more papers. "
    "Use save_notes at the end to save the summaries and the comparison, "
    "unless the user did not ask for notes. "
    "Always cite papers by their arXiv id. "
    "Never invent paper content: only use what the tools returned. "
    "If a tool returns an ERROR, tell the user what went wrong instead of guessing. "
    "In your final answer include the summaries, the comparison table and the saved note path."
)


def build_agent(provider=DEFAULT_PROVIDER):
    return create_agent(
        model=get_llm(provider),
        tools=build_tools(provider),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def run(agent, message, thread_id):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return response["messages"][-1].content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Paper Agent")
    parser.add_argument("request", help="what you want the agent to do")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=PROVIDERS)
    args = parser.parse_args()

    agent = build_agent(args.provider)
    thread_id = str(uuid.uuid4())

    print(f"Provider: {args.provider}\n")
    print(run(agent, args.request, thread_id))
