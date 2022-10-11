from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/age/")
async def age_function():
    return {"Age": "We`ll be predicting in future"}