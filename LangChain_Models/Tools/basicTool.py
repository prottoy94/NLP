#1
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('ipl news')

#print(results)

#2
from langchain_community.tools import ShellTool

shell_tool = ShellTool()

command = 'whoami'

print(f"About to run this command: {command}")
answer = input("Do you want to run it? (yes/no): ").strip().lower()

if answer in ("yes", "y"):
    results = shell_tool.invoke(command)
    print(results)
else:
    print("Command cancelled. Nothing was run.")
