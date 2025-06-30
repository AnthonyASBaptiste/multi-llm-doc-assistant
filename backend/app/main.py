from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI(
    title="Multi-LLM AI Assistant API",
    description="A backend for routing requests to different LLM providers.",
    version="0.1.0",
)

@app.get("/")
def read_root():
    """
    A simple endpoint to confirm the API is running.
    """
    return {"message": "Welcome to your Multi-LLM AI Assistant Backend!"}
