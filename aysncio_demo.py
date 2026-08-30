import asyncio

async def call(i):
    await asyncio.sleep(i)
    return i*2

async def batched(items, size, fn):
    out = []
    for i in range(0,len(items), size):
        chunk = items[i:i+size]
        out += await asyncio.gather(*(fn(x) for x in chunk))
        await asyncio.sleep(0.1)
    return out

async def with_retry(coro_fn, *args,tries=3):
    for attempt in range(tries):
        try:
            return await coro_fn(*args)
        except Exception:
            if attempt == tries - 1:
                await asyncio.sleep(2 ** attempt)
async def main():
    results =  await asyncio.gather(*(call(i) for i in range(5)))
    print(results)
    results = await batched(list(range(10)), size = 5, fn=call)
    print(results)
    try:
        result = await with_retry(call, 2, tries=3)
        print("Retry result:", result)
    except Exception as e:
        print("Final failure:", e)
    

if __name__ == "__main__":
    asyncio.run(main())