# api.py
import os
from fastapi import FastAPI, Request, HTTPException
from detector import detect_prompt_injection

app = FastAPI()
API_KEY = os.getenv("API_KEY", "change-me")

@app.post("/detect")
async def detect(request: Request):
    auth = request.headers.get("authorization")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    data = await request.json()
    prompt = data.get("prompt", "")
    result = detect_prompt_injection(prompt)
    return {"response": result}

# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)