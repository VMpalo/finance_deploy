from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uuid
import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

LANGGRAPH_SERVER_URL = os.getenv("LANGGRAPH_SERVER_URL")

app = FastAPI()


async def send_message_to_langgraph(thread_id: str, user_message: str) -> str:
    """Send a message to LangGraph API and stream back the full assistant response."""
    full_content = ""
    current_event = None

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            url=f"{LANGGRAPH_SERVER_URL}/threads/{thread_id}/runs/stream",
            json={
                "assistant_id": "scout",
                "input": {
                    "messages": [{"role": "human", "content": user_message}]
                },
                "stream_mode": "messages-tuple"
            },
            timeout=60.0
        ) as stream_response:
            async for line in stream_response.aiter_lines():
                if line and line.startswith("data: "):
                    data_content = line[6:]
                    try:
                        message_chunk, _ = eval(data_content)
                        if "content" in message_chunk and message_chunk["content"]:
                            content_piece = message_chunk["content"]
                            full_content += content_piece
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")

    return full_content


@app.get("/", response_class=HTMLResponse)
async def form_get():
    """Show the form without any message."""
    html_content = """
    <html>
        <head>
            <title>Chat with Assistant</title>
        </head>
        <body>
            <h2>Chat with Assistant</h2>
            <form action="/" method="post">
                <textarea name="user_input" rows="4" cols="50" placeholder="Type your message here..."></textarea><br>
                <button type="submit">Send</button>
            </form>
        </body>
    </html>
    """
    return html_content


@app.post("/", response_class=HTMLResponse)
async def form_post(user_input: str = Form(...)):
    """Handle submitted message and show response."""
    thread_id = str(uuid.uuid4())
    assistant_response = await send_message_to_langgraph(thread_id, user_input)

    html_content = f"""
    <html>
        <head>
            <title>Chat with Assistant</title>
        </head>
        <body>
            <h2>Chat with Assistant</h2>
            <p><strong>You:</strong> {user_input}</p>
            <p><strong>Assistant:</strong> {assistant_response}</p>
            <form action="/" method="post">
                <textarea name="user_input" rows="4" cols="50" placeholder="Type your next message here..."></textarea><br>
                <button type="submit">Send</button>
            </form>
        </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
