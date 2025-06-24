# import requests
# import streamlit as st

# API_URL = "http://127.0.0.1:5000/predict"

# st.title("Brain Tumor Prediction")

# uploaded_file = st.file_uploader("Upload a brain MRI image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# if st.button("Predict"):
#     try:
#         # Send image to FastAPI backend
#         files = {"file": uploaded_file.getvalue()}
#         response = requests.post(API_URL, files=files)

#         if response.status_code == 200:
#             result = response.json()["prediction"]
#             confidence = response.json()["confidence"]

#             # Display the result with confidence score
#             st.write(f"**Brain Tumor Prediction: {result}**")
#             st.write(f"**Confidence Score:** {confidence:.2%}")  # Convert to percentage

#         else:
#             st.write("Error:", response.json()["detail"])

#     except Exception as e:
#         st.write("❌ Error:", str(e))



# import requests
# import streamlit as st

# API_URL = "http://127.0.0.1:5000/predict"

# st.title("🧠 Brain Tumor Prediction")

# uploaded_file = st.file_uploader("Upload a brain MRI image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# if st.button("Predict"):
#     try:
#         # Send image to FastAPI backend
#         files = {"file": uploaded_file.getvalue()}
#         response = requests.post(API_URL, files=files)

#         if response.status_code == 200:
#             result = response.json()["prediction"]
#             confidence = response.json()["confidence"]

#             # Display the result with confidence score
#             st.success(f"**Brain Tumor Prediction: {result}**")
#             st.info(f"**Confidence Score:** {confidence:.2%}")  # Convert to percentage

#         else:
#             st.error("❌ Error: " + response.json()["detail"])

#     except Exception as e:
#         st.error("❌ Error: " + str(e))




# import requests
# import streamlit as st

# API_URL = "http://127.0.0.1:5000"

# st.title("🧠 Brain Tumor AI Assistant")

# # ✅ Upload MRI for Prediction
# st.header("🔍 Brain Tumor Prediction")
# uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# if st.button("Predict"):
#     try:
#         files = {"file": uploaded_file.getvalue()}
#         response = requests.post(f"{API_URL}/predict", files=files)

#         if response.status_code == 200:
#             result = response.json()["prediction"]
#             confidence = response.json()["confidence"]
#             ai_explanation = response.json()["ai_explanation"]

#             st.success(f"**Brain Tumor Prediction: {result}**")
#             st.info(f"**Confidence Score:** {confidence:.2%}")
#             st.subheader("📖 AI Explanation & Treatment")
#             st.write(ai_explanation)

#         else:
#             st.error("❌ Error: " + response.json()["detail"])

#     except Exception as e:
#         st.error("❌ Error: " + str(e))

# # ✅ AI Chatbot for Medical Queries
# st.header("💬 AI Medical Chatbot")

# user_input = st.text_input("Ask a medical question about brain tumors:")
# if st.button("Ask AI"):
#     try:
#         response = requests.post(f"{API_URL}/chatbot", json={"query": user_input})
#         if response.status_code == 200:
#             st.success(response.json()["response"])
#         else:
#             st.error("❌ Error: " + response.json()["detail"])
#     except Exception as e:
#         st.error("❌ Error: " + str(e))

import requests
import streamlit as st
from streamlit_option_menu import option_menu

# ✅ Set Page Config
st.set_page_config(page_title="Fruit & Vegetable Disease Detector", layout="centered")

API_URL = "http://127.0.0.1:8000"  # Adjust if your backend is running on a different host/port

# ✅ Sidebar Navigation (Removed Storage Tips and Healthy Recipes)
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["🧪 Detect Quality", "📊 Prediction History", "🤖 AI Assistant"],
        icons=["bug", "bar-chart", "robot"],
        default_index=0
    )

# ✅ Detect Disease
if selected == "🧪 Detect Quality":
    st.title("🧪 Detect Fruit/Vegetable Condition")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Quality"):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_URL}/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    fruit = result["fruit_or_vegetable"]
                    condition = result["condition"]
                    confidence = result["confidence"]
                    tips = result["tips"]  # Get AI-generated tips
                    st.success(f"{fruit} is **{condition}** with {confidence:.2f}% confidence.")
                    st.write(f"💡 Tips: {tips}")

                    if condition.lower() == "rotten":
                        st.warning("⚠️ Not suitable for consumption. Consider composting.")
                    else:
                        st.info("✅ Looks fresh and healthy!")
                else:
                    st.error(response.json().get("detail", "Prediction failed."))
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ✅ Prediction History
elif selected == "📊 Prediction History":
    st.title("📊 Previous Predictions")
    history_response = requests.get(f"{API_URL}/history")
    if history_response.status_code == 200:
        history_data = history_response.json().get("history", [])
        if history_data:
            for idx, entry in enumerate(history_data[::-1]):
                st.write(f"{idx+1}. **Fruit/Vegetable:** {entry['fruit_or_vegetable']} - **Condition:** {entry['condition']} - **Confidence:** {entry['confidence']:.2f}%")
        else:
            st.info("No prediction history yet.")
    else:
        st.error("Unable to fetch history.")

# ✅ AI Assistant
elif selected == "🤖 AI Assistant":
    st.title("🤖 Ask the AI Assistant")
    user_query = st.text_input("Ask something about fruits or veggies:")

    if st.button("Ask"):
        if user_query.strip():
            try:
                response = requests.post(f"{API_URL}/chatbot", json={"query": user_query})

                if response.status_code == 200:
                    st.write("**AI Response:**", response.json().get("response"))
                else:
                    st.error(f"❌ Error from AI API: {response.json().get('detail', 'Unknown error.')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
