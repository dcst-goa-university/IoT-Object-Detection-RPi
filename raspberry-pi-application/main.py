from fastapi import FastAPI

from model import service
from streamio import service
from dto import StatusResponseDTO

app = FastAPI()

@app.get("/")
async def status():
    return {}
