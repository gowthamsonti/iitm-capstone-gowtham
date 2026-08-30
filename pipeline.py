from pydantic import BaseModel

class Question(BaseModel):
    text: str
    
class Answer(BaseModel):
    question: str
    text: str
    cost_usd: float
    retries: int = 0