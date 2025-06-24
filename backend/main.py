# from fastapi import FastAPI, File, UploadFile, HTTPException
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import io
# from PIL import Image

# app = FastAPI()

# # Load the trained model
# try:
#     model = load_model("cnn_model.h5", compile=False)
#     print("✅ Model loaded successfully")
# except Exception as e:
#     print("❌ Error loading model:", str(e))

# # Function to preprocess image
# def preprocess_image(image: Image.Image):
#     image = image.convert("L")  # Convert to grayscale
#     image = image.resize((128, 128))  # Resize to match model input
#     image_array = np.array(image) / 255.0  # Normalize pixel values (0-1)
#     image_array = np.expand_dims(image_array, axis=[0, -1])  # Reshape to (1, 128, 128, 1)
#     return image_array

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read and preprocess image
#         image = Image.open(io.BytesIO(await file.read()))
#         processed_image = preprocess_image(image)

#         # Make prediction
#         prediction = model.predict(processed_image)
#         probability = prediction[0][0]  # Extract probability

#         # Convert probability to "YES" or "NO"
#         threshold = 0.5  # Adjust based on model performance
#         result = "YES" if probability >= threshold else "NO"

#         return {"prediction": result, "confidence": float(probability)}

#     except Exception as e:
#         print("❌ Prediction Error:", str(e))
#         raise HTTPException(status_code=500, detail=str(e))




# from fastapi import FastAPI, File, UploadFile, HTTPException
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import io
# from PIL import Image

# app = FastAPI()

# # Load the trained model
# try:
#     model = load_model("model.h5", compile=False)
#     print("✅ Model loaded successfully")
# except Exception as e:
#     print("❌ Error loading model:", str(e))
#     raise HTTPException(status_code=500, detail="Model loading failed")

# # Function to preprocess image
# # Function to preprocess image
# def preprocess_image(image: Image.Image):
#     image = image.convert("RGB")  # Convert to RGB (3 channels)
#     image = image.resize((128, 128))  # Resize to match model input
#     image_array = np.array(image) / 255.0  # Normalize pixel values (0-1)
#     image_array = np.expand_dims(image_array, axis=0)  # Reshape to (1, 128, 128, 3)
#     return image_array


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read and preprocess image
#         image = Image.open(io.BytesIO(await file.read()))
#         processed_image = preprocess_image(image)

#         # Make prediction
#         prediction = model.predict(processed_image)
#         class_index = np.argmax(prediction)  # Get class with highest probability
#         confidence = float(np.max(prediction))  # Get confidence of prediction

#         # Define class labels (Modify based on your model's classes)
#         class_labels = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]
#         result = class_labels[class_index]

#         return {"prediction": result, "confidence": confidence}

#     except Exception as e:
#         print("❌ Prediction Error:", str(e))
#         raise HTTPException(status_code=500, detail=str(e))


# # Run the backend using: uvicorn backend:app --host 0.0.0.0 --port 5000





# from fastapi import FastAPI, File, UploadFile, HTTPException
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io
# from PIL import Image

# app = FastAPI()

# # Load the trained model (Ensure model.h5 is in the same directory)
# MODEL_PATH = "model.h5"  

# try:
#     model = load_model(MODEL_PATH, compile=False)
#     print("✅ Model loaded successfully")
# except Exception as e:
#     print("❌ Error loading model:", str(e))
#     raise HTTPException(status_code=500, detail="Model loading failed")


# # Function to preprocess image (Updated to match 128×128 input)
# def preprocess_image(image: Image.Image):
#     image = image.convert("RGB")  # Convert to RGB (3 channels)
#     image = image.resize((128, 128))  # Resize to match the model's expected input (128x128)
#     image_array = np.array(image) / 255.0  # Normalize pixel values (0-1)
#     image_array = np.expand_dims(image_array, axis=0)  # Reshape to (1, 128, 128, 3)
#     return image_array


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read and preprocess image
#         image = Image.open(io.BytesIO(await file.read()))
#         processed_image = preprocess_image(image)

#         # Make prediction
#         prediction = model.predict(processed_image)
#         class_index = np.argmax(prediction)  # Get class with highest probability
#         confidence = float(np.max(prediction))  # Get confidence of prediction

#         # Define class labels (Ensure these match the labels used in your Kaggle notebook)
#         class_labels = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]
        
#         if class_index >= len(class_labels):
#             raise ValueError("Class index out of range, check class_labels list.")

#         result = class_labels[class_index]

#         return {"prediction": result, "confidence": confidence}

#     except Exception as e:
#         print("❌ Prediction Error:", str(e))
#         raise HTTPException(status_code=500, detail=str(e))


# # Run the backend using:
# # uvicorn main:app --host 0.0.0.0 --port 5000




# from fastapi import FastAPI, File, UploadFile, HTTPException
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import io
# from PIL import Image
# import requests
# import json

# app = FastAPI()

# # ✅ Load the trained model
# MODEL_PATH = "model.h5"

# try:
#     model = load_model(MODEL_PATH, compile=False)
#     print("✅ Model loaded successfully")
# except Exception as e:
#     print("❌ Error loading model:", str(e))
#     raise HTTPException(status_code=500, detail="Model loading failed")

# # ✅ Set up Gemini API
# GEMINI_API_KEY = "AIzaSyDdifJhrztNdBYGKGWM1xDtQr3vP2GSTds"
# GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# # ✅ Function to preprocess image (Fixed for 128x128 input)
# def preprocess_image(image: Image.Image):
#     image = image.convert("RGB")  
#     image = image.resize((128, 128))  # Ensure model gets correct input size
#     image_array = np.array(image) / 255.0  
#     image_array = np.expand_dims(image_array, axis=0)  
#     return image_array

# # ✅ Function to get AI-generated response from Gemini (Fixed for Chatbot!)
# def get_gemini_response(prompt):
#     headers = {"Content-Type": "application/json"}
#     data = {"contents": [{"parts": [{"text": prompt}]}]}

#     try:
#         response = requests.post(GEMINI_API_URL, headers=headers, json=data)
#         response.raise_for_status()
#         full_response = response.json()

#         # ✅ Fix: Properly extract text from API response
#         if "candidates" in full_response and full_response["candidates"]:
#             parts = full_response["candidates"][0]["content"]["parts"]
#             if isinstance(parts, list):  
#                 return " ".join([part["text"] for part in parts])  
#             elif isinstance(parts, dict):  
#                 return parts.get("text", "No response available.")  
#             else:
#                 return "No response available."
#         else:
#             return "AI response not available."
        
#     except Exception as e:
#         print("❌ Gemini API Error:", str(e))
#         return "AI response unavailable at the moment."

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         image = Image.open(io.BytesIO(await file.read()))
#         processed_image = preprocess_image(image)

#         prediction = model.predict(processed_image)
#         class_index = np.argmax(prediction)
#         confidence = float(np.max(prediction))

#         class_labels = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]

#         if class_index >= len(class_labels):
#             raise ValueError("Class index out of range, check class_labels list.")

#         result = class_labels[class_index]

#         # ✅ Get AI explanation for tumor type
#         ai_explanation = get_gemini_response(f"Explain {result} and suggest treatments.")

#         return {
#             "prediction": result,
#             "confidence": confidence,
#             "ai_explanation": ai_explanation,
#         }

#     except Exception as e:
#         print("❌ Prediction Error:", str(e))
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chatbot")
# async def chatbot(data: dict):
#     try:
#         query = data.get("query", "")
#         if not query:
#             raise ValueError("Query cannot be empty.")

#         ai_response = get_gemini_response(f"Answer this medical question: {query}")
#         return {"response": ai_response}

#     except Exception as e:
#         print("❌ Chatbot Error:", str(e))
#         raise HTTPException(status_code=500, detail="Chatbot is unavailable at the moment.")


from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import google.generativeai as genai
import requests
import base64
import matplotlib.pyplot as plt


app = FastAPI()

# ✅ Gemini API Setup
GEMINI_API_KEY = "AIzaSyDdifJhrztNdBYGKGWM1xDtQr3vP2GSTds"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

genai.configure(api_key="AIzaSyDdifJhrztNdBYGKGWM1xDtQr3vP2GSTds")

# Load the model
model = load_model("fruit_model.h5", compile=False)

FRUIT_VEG_CLASSES = {
    0: "Apple_Healthy", 1: "Apple_Rotten",
    2: "Banana_Healthy", 3: "Banana_Rotten",
    4: "Bellpepper_Healthy", 5: "Bellpepper_Rotten",
    6: "Carrot_Healthy", 7: "Carrot_Rotten",
    8: "Cucumber_Healthy", 9: "Cucumber_Rotten",
    10: "Grape_Healthy", 11: "Grape_Rotten",
    12: "Guava_Healthy", 13: "Guava_Rotten",
    14: "Jujube_Healthy", 15: "Jujube_Rotten",
    16: "Mango_Healthy", 17: "Mango_Rotten",
    18: "Orange_Healthy", 19: "Orange_Rotten",
    20: "Pomegranate_Healthy", 21: "Pomegranate_Rotten",
    22: "Potato_Healthy", 23: "Potato_Rotten",
    24: "Strawberry_Healthy", 25: "Strawberry_Rotten",
    26: "Tomato_Healthy", 27: "Tomato_Rotten",
}

prediction_history = []

def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((224, 224))
    array = np.expand_dims(np.array(image) / 255.0, axis=0)
    return array

def generate_tips(fruit, condition):
    if condition.lower() == "rotten":
        prompt = f"""
        The user has a rotten {fruit.lower()}. Provide:
        1. Two short and clear tips on how to deal with rotten produce (e.g., composting, disposal).
        2. One suggestion for preventing spoilage in the future.
        3. One fun fact about dealing with food waste or rotten produce.
        """
    else:
        prompt = f"""
        The user has a healthy {fruit.lower()}. Provide:
        1. Two short and clear storage tips to keep it fresh.
        2. One simple healthy recipe.
        3. One fun nutritional fact about this fruit or vegetable.
        """

    model = genai.GenerativeModel("gemini-1.5-flash")  # You can switch to "gemini-1.0-pro" if needed
    response = model.generate_content(prompt)
    return response.text



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        label = FRUIT_VEG_CLASSES[pred_class]
        fruit, condition = label.split("_")

        tips = generate_tips(fruit, condition)

        result = {
            "fruit_or_vegetable": fruit,
            "condition": condition,
            "confidence": round((confidence * 100)-2, 2),
            "tips": tips,
            "img_path": file.filename
        }

        prediction_history.append(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    return {"history": prediction_history}

# ✅ Chatbot Endpoint
@app.post("/chatbot")
async def chatbot(query: dict):
    try:
        question = query.get("query", "")
        if not question:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Sending query to Gemini API for generating response
        response = requests.post(GEMINI_API_URL, json={"contents": [{"parts": [{"text": question}]}]})
        
        # Check for successful response
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error in Gemini API response")

        # Parse the Gemini API response
        response_data = response.json()
        ai_response = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response available.")

        # Return the AI response
        return {"response": ai_response}
    
    except Exception as e:
        print("❌ Chatbot Error:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
