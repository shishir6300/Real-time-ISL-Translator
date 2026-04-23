from flask import Flask, Response, request, jsonify, render_template, send_file
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import enchant
from gtts import gTTS
import os
import traceback
import time
from googletrans import Translator
import random
import string

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model("C:\\Users\\Tejaswini\\Downloads\\model.h5")


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define the alphabet
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize the English dictionary
d = enchant.Dict("en_US")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Current word variable
current_word = ""

def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    temp_list = np.array([[x - base_x, y - base_y] for x, y in landmark_list]).flatten()
    max_value = max(abs(temp_list))
    return temp_list / max_value if max_value > 0 else temp_list

def get_suggestions(word):
    if len(word) < 2:
        return ["Keep typing..."]
    suggestions = d.suggest(word)
    if not suggestions:
        suggestions = ["No suggestions"]
    if word not in suggestions:
        suggestions.insert(0, word)
    return suggestions[:4]

def select_nearest_hands(hands_list):
    def distance_to_camera(landmarks):
        palm = landmarks[0]
        return palm[1]  # Lower y is closer (top of frame)

    sorted_hands = sorted(hands_list, key=distance_to_camera)
    return sorted_hands[:2]  # Take up to 2 hands (left and right)

def generate_frames():
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks_list = [calc_landmark_list(frame, hl) for hl in results.multi_hand_landmarks]
                selected_hands = select_nearest_hands(hand_landmarks_list)

                for hand_landmarks in selected_hands:
                    mp_drawing.draw_landmarks(
                        frame,
                        mp.solutions.hands.Hands().process(frame).multi_hand_landmarks[hand_landmarks_list.index(hand_landmarks)],
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/predict", methods=["POST"])
def predict():
    global current_word

    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({"detected_class": "-", "current_word": current_word, "suggestions": get_suggestions(current_word)})

        hand_landmarks_list = [calc_landmark_list(image, hl) for hl in results.multi_hand_landmarks]
        selected_hands = select_nearest_hands(hand_landmarks_list)

        detected_letter = "-"
        for landmarks in selected_hands:
            processed = pre_process_landmark(landmarks)
            predictions = model.predict(pd.DataFrame([processed]), verbose=0)
            detected_letter = alphabet[np.argmax(predictions)]

            if detected_letter != "-":
                current_word += detected_letter.lower()

        return jsonify({
            "detected_class": detected_letter,
            "current_word": current_word,
            "suggestions": get_suggestions(current_word)
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear():
    global current_word
    current_word = ""
    return jsonify({"status": "cleared"})

@app.route("/suggestions", methods=["POST"])
def suggestions():
    data = request.get_json()
    partial_word = data.get("current_word", "")
    return jsonify({"suggestions": get_suggestions(partial_word)})

# Folder to store generated mp3 files
AUDIO_FOLDER = "generated_audio"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Variable to track the current audio file
current_audio_file = None

# Full language code to gTTS language map
LANGUAGE_MAP = {
    "en": "en",        # English
    "en-US": "en",     # US English (fallback to plain English)
    "hi": "hi",        # Hindi
    "hi-IN": "hi",     # Hindi (India)
    "te": "te",        # Telugu
    "te-IN": "te",     # Telugu (India)
    "bn": "bn",        # Bengali
    "bn-IN": "bn",     # Bengali (India)
    "ta": "ta",        # Tamil
    "ta-IN": "ta",     # Tamil (India)
    "kn": "kn",        # Kannada
    "kn-IN": "kn",     # Kannada (India)
    "ml": "ml",        # Malayalam
    "ml-IN": "ml",     # Malayalam (India)
    "pa": "pa",        # Punjabi
    "pa-IN": "pa",     # Punjabi (India)
}

@app.route("/speak", methods=["POST"])
def speak():
    global current_audio_file

    try:
        data = request.get_json()
        text = data.get("text", "")
        language = data.get("language", "en")

        # Map to gTTS supported code
        language = LANGUAGE_MAP.get(language, language)

        # Supported languages for both translation and text-to-speech
        supported_languages = set(LANGUAGE_MAP.keys())

        if language not in supported_languages:
            return jsonify({"error": f"Unsupported language: {language}. Supported languages: {list(supported_languages)}"}), 400

        # Translate the text to the selected language
        translator = Translator()
        translated_text = translator.translate(text, dest=language).text

        # Generate a unique filename to avoid caching issues
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        timestamp = int(time.time())
        filename = f"speech_{timestamp}_{random_str}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)

        # Generate speech from translated text
        tts = gTTS(text=translated_text, lang=language)
        tts.save(filepath)

        # Delete old audio file to free space
        if current_audio_file and os.path.exists(current_audio_file):
            os.remove(current_audio_file)

        # Update the current audio file tracker
        current_audio_file = filepath

        # Return the new audio file with a timestamp to prevent caching
        return jsonify({
            "translated_text": translated_text,
            "audio_url": f"/current_audio?t={timestamp}"
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)})

@app.route("/current_audio")
def get_current_audio():
    if not current_audio_file or not os.path.exists(current_audio_file):
        return jsonify({"error": "No audio file generated yet."}), 404

    return send_file(current_audio_file, mimetype="audio/mpeg", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)