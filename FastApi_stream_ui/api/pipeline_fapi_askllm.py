from openai import AsyncOpenAI
from .pipeline_model import Question, Answer
import logging,json,time
import asyncio

client = AsyncOpenAI()

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "msg": record.getMessage(),
            })
    
log = logging.getLogger("pipeline")
log.setLevel(logging.INFO)
h = logging.StreamHandler(); h.setFormatter(JsonFormatter())
log.addHandler(h)



async def ask_llm(q: Question):
    resp = await client.chat.completions.create(
        model = "gpt-5.4-mini",
        messages = [{"role":"user", "content":q.question}]
    )

    ans =  Answer(
        question= q.question,
        text = resp.choices[0].message.content,
        cost_usd= 0.0001
    )
    log.info(f"asked: {q.question[:40]}")
    return ans


async def ask_llm_with_retry(q: Question, tries: int = 3) -> Answer:
    for attempt in range(tries):
        try:
            ans = await ask_llm(q)
            ans.retries = attempt
            return ans
        except Exception:
            if attempt == tries -1: raise
            await asyncio.sleep(2 ** attempt) #1s, 2s, 4s 

async def main():
    q = Question(text="What is RAG in one sentence?")
    ans = await ask_llm_with_retry(q)
    print(ans)

if __name__ == "__main__":
    asyncio.run(main())
