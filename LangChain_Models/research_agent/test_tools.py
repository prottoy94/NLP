import re
import sys
import time

from tools import build_tools


def run_tool(tools, title, tool_name, tool_input):
    print("\n" + "=" * 60)
    print(f"{title}  ->  {tool_name}")
    print("=" * 60)

    start_time = time.time()
    output = tools[tool_name].invoke(tool_input)
    end_time = time.time()

    print(output)
    print(f"[{end_time - start_time:.1f}s]")

    if output.startswith("ERROR"):
        print(f"❌ FAILED at {tool_name}")
        sys.exit(1)

    return output


def main():
    if len(sys.argv) > 1:
        provider = sys.argv[1]
    else:
        provider = "gemini"
    print(f"Provider: {provider}")

    tools = {}
    for tool in build_tools(provider):
        tools[tool.name] = tool

    search_result = run_tool(
        tools,
        "Step 1: Search arXiv",
        "search_arxiv",
        {"query": "AI-generated text detection", "max_results": 3, "sort": "recent"},
    )

    paper_ids = re.findall(r"^\d+\. \[([^\]]+)\]", search_result, flags=re.MULTILINE)
    if len(paper_ids) < 2:
        print("❌ Expected at least 2 papers from the search")
        sys.exit(1)

    first_paper = paper_ids[0]

    run_tool(tools, "Step 2: Load paper", "load_paper", {"arxiv_id": first_paper})
    run_tool(tools, "Step 2b: Load again (cached)", "load_paper", {"arxiv_id": first_paper})

    run_tool(
        tools,
        "Step 3: Summarize paper",
        "summarize_paper",
        {"arxiv_id": first_paper, "style": "Technical"},
    )

    run_tool(
        tools,
        "Step 4: Ask the paper",
        "ask_paper",
        {"arxiv_id": first_paper, "question": "Which datasets or benchmarks were used for evaluation?"},
    )
    run_tool(
        tools,
        "Step 4b: Question NOT in the paper",
        "ask_paper",
        {"arxiv_id": first_paper, "question": "What is the capital of Australia?"},
    )

    table = run_tool(
        tools,
        "Step 5: Compare papers",
        "compare_papers",
        {"arxiv_ids": paper_ids[:2]},
    )

    run_tool(
        tools,
        "Step 6: Save notes",
        "save_notes",
        {"title": "Milestone 1 test - AI text detection", "markdown": table},
    )

    print("\n✅ ALL 6 TOOLS WORKED")


if __name__ == "__main__":
    main()
