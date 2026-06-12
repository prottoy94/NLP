from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, EmailStr, Field
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

#schema
json_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "description": "A list of the main themes or topics discussed in the review in a list",
      "items": {
        "type": "string"
      }
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review, capturing the main points in one or two sentences."
    },
    "sentiment": {
      "type": "string",
      "description": "The overall sentiment of the review, either positive, negative, or neutral.",
      "enum": ["positive", "negative"]
    },
    "pros": {
      "type": ["array", "null"],
      "description": "A list of the positive aspects mentioned in the review, if any. If there are no pros, this should be an empty list or null.",
      "items": {
        "type": "string"
      }
    },
    "cons": {
      "type": ["array", "null"],
      "description": "A list of the negative aspects mentioned in the review, if any. If there are no cons, this should be an empty list or null.",
      "items": {
        "type": "string"
      }
    },
    "reviewer_name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer who wrote the review. If the name is not mentioned in the review, this should be null."
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  
        temperature=0.9,
        max_output_tokens=2048,    
        google_api_key=api_key
    )
structured_model=model.with_structured_output(json_schema)
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

Reviewd by Prottoy Roy
""")

print(result)
