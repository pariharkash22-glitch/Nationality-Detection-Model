import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import webcolors

# Function to get color names
def get_color_name(rgb_triplet):
    try:
        return webcolors.rgb_to_name(rgb_triplet)
    except ValueError:
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - rgb_triplet[0]) ** 2
            gd = (g_c - rgb_triplet[1]) ** 2
            bd = (b_c - rgb_triplet[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]

def get_dress_color(img_array, face_region):
    x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
    # ROI: Sampling the area below the chin
    dress_roi = img_array[y+h : y+h+int(h*0.8), x:x+w]
    if dress_roi.size == 0: 
        return (100, 100, 100), "Unknown"
    avg_color = np.average(np.average(dress_roi, axis=0), axis=0).astype(int)
    return tuple(avg_color), get_color_name(tuple(avg_color))

# --- UI Setup ---
st.set_page_config(page_title="Nationality AI", layout="wide")
st.title("🌍 Nationality & Emotion Detector")

uploaded_file = st.file_uploader("Upload a CLEAR portrait photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Load and Resize Image to prevent OOM (Out of Memory)
    image = Image.open(uploaded_file)
    
    # Resize to a max width of 800px while keeping aspect ratio
    max_size = 800
    ratio = max_size / max(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)
    
    img_array = np.array(image.convert('RGB'))
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🖼️ Input Preview")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 Analysis Results")
        
        try:
            with st.spinner('AI is analyzing... (This may take a moment on first run)'):
                # Use a lighter model (Fast) to save memory
                analysis = DeepFace.analyze(
                    img_array, 
                    actions=['race', 'age', 'emotion'], 
                    enforce_detection=True,
                    detector_backend='opencv' # More memory efficient backend
                )
                
            res = analysis[0]
            race = res['dominant_race']
            emotion = res['dominant_emotion']
            age = res['age']

            st.success(f"**Detected Nationality/Race:** {race.capitalize()}")
            st.info(f"**Primary Emotion:** {emotion.title()}")

            # Conditional Logic for Outputs
            if race == 'indian':
                st.write(f"**Estimated Age:** {age}")
                rgb, c_name = get_dress_color(img_array, res['region'])
                st.write(f"**Dress Color:** {c_name.title()}")
                st.color_picker("Detected Shade", value='#%02x%02x%02x' % rgb, disabled=True)

            elif race in ['white', 'latino hispanic']:
                st.write(f"**Estimated Age:** {age}")

            elif race == 'black':
                rgb, c_name = get_dress_color(img_array, res['region'])
                st.write(f"**Dress Color:** {c_name.title()}")
                st.color_picker("Detected Shade", value='#%02x%02x%02x' % rgb, disabled=True)

        except ValueError:
            st.error("❌ **No face detected.** Please ensure the face is clear and looking at the camera.")
        except Exception as e:
            st.error(f"⚠️ Memory or Processing Error: {str(e)[:100]}...")
            st.warning("Try uploading a smaller file or closing other browser tabs.")