# Deepfake-Defaker
Deepfake Defaker

# Requirements
PyTorch:        pip install torch
Torchvision:    pip install torchvision
Uvicorn:        pip install uvicorn
FastAPI:        pip install fastAPI


# Kaggle Steps: 
Users must get their authentication key from Kaggle.com. As the Kaggle API will not run without your key. 
Next step is users must use the Van Gogh Paintings dataset from following Kaggle user: ipythonx/van-gogh-paintings (ipythonx is the users name, and Van-gogh-paintings is name of dataset used in this project)


# Run FastAPI server
To run the server you must run the bash command `uvicorn app.defaker-test:app --host 127.0.0.1 --port 5000 --reload`.
This will run the server using FastAPI and ensure the Python file can connect to the JavaScript file via an endpoint.
