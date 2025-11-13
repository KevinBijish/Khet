import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert to percentage
    return result_index, confidence

st.set_page_config(page_title="Khet Sahayak", layout="wide")

# --- CSS: Everything white background, modern style, all main elements themed ---
st.markdown("""
    <style>
    html, body, .main {background: #fff!important;}
    body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #222;}
    .disease-section {
        background: #fff; border-radius: 18px; padding: 50px 25px; box-shadow: 0 10px 30px rgba(44,68,116,0.08);
        margin:40px auto 24px auto; max-width:920px;
    }
    .disease-header {text-align:center; margin-bottom:36px;}
    .disease-header h1 {font-size:34px; color:#18683A; font-weight:700; letter-spacing:0.5px;}
    .about-content p, .desc-p {font-size:18px; line-height:1.7; color:#444;}
    .main-title {font-size:26px; font-weight:bold; color:#249f56; text-align:center; 
                 margin:24px 0 10px 0; letter-spacing:0.3px;}
    .footer {background:#fff; color:#999; text-align:center; padding:18px;}
    .footer p {margin:0; font-size:15px;}

    .topbar {width:100%; background:#fff; box-shadow:0 2px 10px rgba(0,0,0,0.05); z-index:100; margin-bottom:14px;}
    .navbar-inner {height:60px; display:flex; align-items:center; justify-content:space-between; max-width:950px; margin:0 auto; padding:0 18px;}
    .language-form label {margin-right:10px; font-size:15px;}

    @media (max-width: 650px) {
        .disease-section {padding:15px 2px;}
        .disease-header h1 {font-size:22px;}
        .main-title {font-size:18px;}
        .navbar-inner {height:auto; flex-direction:column; gap:4px;}
    }
    </style>
""", unsafe_allow_html=True)

# Topbar: Large brand + language picker
st.markdown("""
<div class="topbar">
  <div class="navbar-inner">
    <span class="main-title">Khet Sahayak · Disease Recognition</span>
    <div class="language-form"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Language selector and UI labels/strings
langs = {"English": "en", "हिन्दी": "hi", "ਪੰਜਾਬੀ": "pa"}
lang = st.selectbox("Language / भाषा / ਭਾਸ਼ਾ", options=list(langs.keys()), index=0, key="langbox")
cur_lang = langs[lang]

ui_text = {
    "en": {
        "title": "Disease Identification",
        "desc": "Identify crop diseases early with advanced image recognition technology.\nUpload an image below for an instant diagnosis.",
        "choose_img": "Choose an Image:",
        "show_img": "Show Image",
        "predict_btn": "Predict",
        "prediction": "Our Prediction",
        "result_prefix": "Model predicts: ",
        "confidence": "Confidence: ",
        "footer": "© 2025 Khet Sahayak. All rights reserved."
    },
    "hi": {
        "title": "रोग की पहचान",
        "desc": "आधुनिक इमेज प्रोसेसिंग से फसल की बीमारियों को जल्दी पहचानें। नीचे छवि अपलोड करें और त्वरित परिणाम पाएं।",
        "choose_img": "छवि चुनें:",
        "show_img": "छवि दिखाएं",
        "predict_btn": "पूर्वानुमान लगाएं",
        "prediction": "हमारा पूर्वानुमान",
        "result_prefix": "मॉडल का पूर्वानुमान: ",
        "confidence": "विश्वास स्तर: ",
        "footer": "© 2025 खेत सहायक. सर्वाधिकार सुरक्षित."
    },
    "pa": {
        "title": "ਬੀਮਾਰੀ ਪਛਾਣ",
        "desc": "ਤਕਨੀਕੀ ਚਿੱਤਰ ਪਛਾਣ ਰਾਹੀਂ ਫਸਲ ਦੀਆਂ ਬੀਮਾਰੀਆਂ ਜ਼ਲਦੀ ਪਛਾਣੋ। ਥੱਲੇ ਤਸਵੀਰ ਅੱਪਲੋਡ ਕਰੋ ਤੇ ਤੁਰੰਤ ਨਤੀਜਾ ਲਵੋ।",
        "choose_img": "ਚਿੱਤਰ ਚੁਣੋ:",
        "show_img": "ਚਿੱਤਰ ਵੇਖਾਓ",
        "predict_btn": "ਅੰਦਾਜ਼ਾ ਲਵੋ",
        "prediction": "ਸਾਡਾ ਅੰਦਾਜ਼ਾ",
        "result_prefix": "ਮਾਡਲ ਅਨੁਸਾਰ: ",
        "confidence": "ਵਿਸ਼ਵਾਸ ਪੱਧਰ: ",
        "footer": "© 2025 ਖੇਤ ਸਹਾਇਕ. ਸਰਬੱਤ ਹੱਕ ਰਾਖਵੇਂ ਹਨ."
    },
}
labels = ui_text[cur_lang]

st.markdown(f"""
<div class="disease-section">
    <div class="disease-header">
        <span style="font-size:36px;">🦠</span>
        <h1>{labels['title']}</h1>
    </div>
    <div class="about-content"><p class="desc-p">{labels['desc']}</p></div>
    <div style="margin-top:36px;">
""", unsafe_allow_html=True)

test_image = st.file_uploader(labels['choose_img'])
if test_image is not None:
    if st.button(labels['show_img']):
        st.image(test_image, width=400, use_column_width=True)
    if st.button(labels['predict_btn']):
        st.snow()
        st.write(labels['prediction'])
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        result_index, confidence = model_prediction(test_image)
        st.success(f"{labels['result_prefix']}{class_name[result_index]}")
        st.info(f"{labels['confidence']}{confidence:.2f}%")
        
st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown(
    f'<div class="footer"><p>{labels["footer"]}</p></div>',
    unsafe_allow_html=True
)
