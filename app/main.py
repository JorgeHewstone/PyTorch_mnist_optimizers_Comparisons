import torch
import torch.nn as nn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
import numpy as np
import base64
import re
from io import BytesIO
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# Importar nuestra clase de modelo desde la carpeta src
import sys
import os
# Añadimos la carpeta 'src' al path para poder importar 'model.py'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SimpleNN 

# --- Configuración de la App ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Carga de Modelos ---
# (Cargamos los modelos al iniciar la app)
DEVICE = "cpu" # Usamos CPU para inferencia
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Lista de modelos disponibles (basados en los archivos .pth)
available_models = {}
print("Cargando modelos...")
for f in os.listdir(MODEL_DIR):
    if f.endswith(".pth"):
        model_name = f.replace("model_", "").replace(".pth", "").upper()
        
        model_path = os.path.join(MODEL_DIR, f)
        model = SimpleNN()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Poner en modo evaluación
        
        available_models[model_name] = model
        print(f"Modelo cargado: {model_name} desde {f}")

# --- Transformación para la Inferencia ---
# ¡Debe ser la MISMA normalización que en el entrenamiento!
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- Clases Pydantic para la API ---
class PredictionRequest(BaseModel):
    model_name: str  # ej: "ADAM"
    image_data: str  # Imagen en formato base64 data URL

# --- Rutas de la API ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal con el canvas."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_names": list(available_models.keys()) # Pasamos la lista al frontend
    })

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Recibe un dibujo y un modelo, y devuelve una predicción."""
    
    model = available_models.get(request.model_name)
    if not model:
        return {"error": "Modelo no encontrado"}

    try:
        # 1. Decodificar la imagen Base64
        # (El frontend envía "data:image/png;base64,iVBOR...")
        img_data = re.sub('^data:image/.+;base64,', '', request.image_data)
        img_bytes = base64.b64decode(img_data)
        
        # 2. Convertir a imagen PIL y pre-procesar
        img_pil = Image.open(BytesIO(img_bytes)).convert('L') # Convertir a escala de grises
        
        # 3. Redimensionar a 28x28 (MNIST size)
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 4. Convertir a Tensor y Normalizar
        img_pil = ImageOps.invert(img_pil)
        tensor = inference_transform(img_pil).unsqueeze(0) # Añadir batch dimension (1, 1, 28, 28)

        # 5. Realizar la predicción
        with torch.no_grad():
            output = model(tensor.to(DEVICE))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item() * 100

        return {
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%"
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Para ejecutar la app:
    # 1. Asegúrate de estar en la carpeta 'app/'
    # 2. Ejecuta: uvicorn main:app --reload
    print("Inicia el servidor con: uvicorn main:app --reload")