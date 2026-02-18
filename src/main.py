import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents import get_workflow

app = FastAPI(title="Sentinel Clinical API")

class AuditRequest(BaseModel):
    history: str

@app.get("/")
async def root():
    return {"status": "Sentinel API is Online"}

@app.post("/audit")
async def run_audit(request: AuditRequest):
    try:
        workflow = get_workflow()
        result = workflow.invoke({"history": request.history, "biomarkers": {}, "relevant_guidelines": [], "report": ""})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)