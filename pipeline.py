from pydantic import BaseModel, Field

class Question(BaseModel):
    text: str

class Question2(BaseModel):
    question: str = Field(..., min_length=1 )
    
class Answer(BaseModel):
    question: str
    text: str
    cost_usd: float
    retries: int = 0