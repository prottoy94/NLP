from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")
    
def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput, # args_schema is where MultiplyInput gets attached to the tool.
    func=multiply
)

result = multiply_tool.invoke({"a": 3, "b": 5})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
