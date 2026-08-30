from pydantic import BaseModel
from typing import Optional
import asyncio

class User(BaseModel):
    name:str
    age:int
    email:Optional[str]=None

class ExpectedAnswer(BaseModel):
    summary:str
    confidence:float
    sources:list[str]
raw =  '{"summary": "...", "confidence": 0.9, "sources": ["d1"]}'

async def hello(name):
    await asyncio.sleep(1.0)
    return f"hi,{name}"

print(asyncio.run(hello("Asha")))
parsed =  ExpectedAnswer.model_validate_json(raw)
print(parsed)
u = User(name="Asha", age="12")
print(u.model_dump())

