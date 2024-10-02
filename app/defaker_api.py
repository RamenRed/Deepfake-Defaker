from fastapi import FastAPI
import app.defaker_backend as df

app = FastAPI()

# =================================================================================
#                         Endpoints for Helper Functions
# =================================================================================
@app.get("/endpoint/template")
async def api_function_template():
    return "This is an example of an API endpoint"


# =================================================================================
#                           Endpoints for GAN classes
# =================================================================================
