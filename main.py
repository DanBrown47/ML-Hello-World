from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/age/{age_val}")
async def age_function(age_val:int):
    return {"Age": age_val}
