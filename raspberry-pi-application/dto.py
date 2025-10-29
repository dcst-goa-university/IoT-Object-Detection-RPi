from pydantic import BaseModel
from fastapi import status

class StatusResponseDTO(BaseModel):
    server: status
    model: status
    
