from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
# ElevenLabs
import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
trafficModel = YOLO("best.pt")
carModel = YOLO("car-detection.pt")

load_dotenv()
client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

@app.route("/traffic", methods=['POST'])
def predictTraffic():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Read image data into OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Run inference
    results = trafficModel(img)[0]

    # Define vars
    class_names = results.names
    predictions = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        prediction = {
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, "unknown"),
                "confidence": float(box.conf[0]),
                "bbox": [float(coord) for coord in box.xyxy[0]]
        }
        predictions.append(prediction)

    return jsonify({"trafficPrediction": predictions})

@app.route("/car", methods=['POST'])
def predictCar():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Read image data into OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Run inference
    results = carModel(img)[0]
    class_names = results.names
    predictions = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        prediction = {
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, "unknown"),
                "confidence": float(box.conf[0]),
                "bbox": [float(coord) for coord in box.xyxy[0]]
        }
        predictions.append(prediction)

    print(predictions)
    return jsonify({"carPrediction": predictions})

# TTS is not tested
@app.route("/tts", methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        audio = client.text_to_speech.convert(
            text=text,
            voice_id="56AoDkrOh6qfVPDXZ7Pt",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        return Response(audio, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# == Example Run ==
#   "TrafficPrediction": [
#     {
#       "bbox": [
#         609.4138793945312,
#         246.30369567871094,
#         665.3598022460938,
#         322.8956604003906
#       ],
#       "class_id": 7,
#       "class_name": "pedunknown",
#       "confidence": 0.9992734789848328
#     },
#     {
#       "bbox": [
#         678.4158325195312,
#         239.96865844726562,
#         756.5810546875,
#         320.4295654296875
#       ],
#       "class_id": 5,
#       "class_name": "pedred",
#       "confidence": 0.99146568775177
#     },
#     {
#       "bbox": [
#         1052.14794921875,
#         355.3636474609375,
#         1066.4017333984375,
#         379.6042785644531
#       ],
#       "class_id": 4,
#       "class_name": "unknown",
#       "confidence": 0.9668723940849304
#     },
#     {
#       "bbox": [
#         358.2610168457031,
#         430.9151916503906,
#         374.36370849609375,
#         444.66357421875
#       ],
#       "class_id": 7,
#       "class_name": "pedunknown",
#       "confidence": 0.9512879252433777
#     },
#     {
#       "bbox": [
#         870.4915161132812,
#         419.3833312988281,
#         876.0985717773438,
#         432.7700500488281
#       ],
#       "class_id": 4,
#       "class_name": "unknown",
#       "confidence": 0.8823534250259399
#     },
#     {
#       "bbox": [
#         845.417236328125,
#         353.7409362792969,
#         860.5087890625,
#         381.13433837890625
#       ],
#       "class_id": 4,
#       "class_name": "unknown",
#       "confidence": 0.8536435961723328
#     },
#     {
#       "bbox": [
#         1065.4102783203125,
#         429.0877990722656,
#         1083.0474853515625,
#         444.5091552734375
#       ],
#       "class_id": 7,
#       "class_name": "pedunknown",
#       "confidence": 0.838220477104187
#     },
#     {
#       "bbox": [
#         1054.084716796875,
#         430.0879821777344,
#         1064.530517578125,
#         445.2373046875
#       ],
#       "class_id": 7,
#       "class_name": "pedunknown",
#       "confidence": 0.8105863928794861
#     },
#     {
#       "bbox": [
#         845.9163818359375,
#         355.0885925292969,
#         856.8585815429688,
#         381.75384521484375
#       ],
#       "class_id": 4,
#       "class_name": "unknown",
#       "confidence": 0.5940747261047363
#     },
#     {
#       "bbox": [
#         854.528564453125,
#         354.2474365234375,
#         862.5054321289062,
#         380.1994323730469
#       ],
#       "class_id": 4,
#       "class_name": "unknown",
#       "confidence": 0.4625633656978607
#     }
#   ]
# }
