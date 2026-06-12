from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    cgpa: float= Field(..., gt=0.0, lt=4.0, default=0.0, description="CGPA must be between 0.0 and 4.0. If the cgpa is above 3.9, the student is considered excellent.")
    
new_student = {"name": "Alice", "cgpa": 3.5}
new_student = {"age": "32", "name": "Bob", "cgpa": 3.8}
new_student = {"email": "bob@example.com", "name": "Bob", "cgpa": 3.9}
new_student=Student(**new_student)

print(new_student)