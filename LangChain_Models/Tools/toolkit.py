from langchain_core.tools import tool


# Custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


class MathToolkit:
    def get_tools(self):
        return [add, multiply]
    
toolkit= MathToolkit()
tools =toolkit.get_tools()

for tool in tools:
    result = tool.invoke({"a": 3, "b": 5})
    print(f"Result of {tool.name}: {result}")
    print(f"Description: {tool.description}")
    print(f"Arguments: {tool.args}")
    print(f"Arguments Schema: {tool.args_schema.model_json_schema()}")