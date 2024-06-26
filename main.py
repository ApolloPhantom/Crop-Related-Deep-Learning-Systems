from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class LoginRequest(BaseModel):
    aadhar: str
    password: str

@app.post("/login")
def login(request: LoginRequest):
    # Here you should add your actual authentication logic
    if request.aadhar == "100" and request.password == "password":
        return {"message": "Login successful"}
    else:
        raise HTTPException(status_code=400, detail="Invalid credentials")

@app.get("/")
def read_root():
    return {"message": "Hello World"}
