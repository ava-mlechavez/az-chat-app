import asyncio


async def collect_and_stream(response):
    content = ""

    queue = asyncio.Queue()

    async def collect_content():
        nonlocal content
        async for chunk in response:
            if chunk.content:
                content += chunk.content
                await queue.put(chunk.content)

        await queue.put(None)

    async def stream_response():
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    await collect_content()

    return stream_response(), content
