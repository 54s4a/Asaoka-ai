from fastapi import FastAPI

app = FastAPI(title="Asaoka AI")

@app.get("/")
def root():
    return {"status": "ok", "message": "Asaoka AI backend running successfully."}
