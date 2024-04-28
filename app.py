from flask import Flask, jsonify, request
import cv2
import numpy as np
import math
from super_gradients.training import models
from flask_cors import CORS
import base64
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app)

Model_arch = 'yolo_nas_m'
best_model = models.get(
    Model_arch,
    num_classes=4,
    checkpoint_path='NewModel.pth'
)

names = ["balloon", "bird", "drone", "plane"]

def detect_objects(frame):
    # Perform object detection on the frame
    prediction = best_model.predict(frame, conf=0.35)
    detected_objects = []
    bbox_xyxys = prediction.prediction.bboxes_xyxy.tolist()
    confidences = prediction.prediction.confidence
    labels = prediction.prediction.labels.tolist()
    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        classname = int(cls)
        class_name = names[classname]
        conf = math.ceil((confidence*100))/100
        if conf > 0.4:
            if class_name == 'drone':
                if conf > 0.5:
                    detected_objects.append('Drone detected!')
    return detected_objects

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Receive frame data from frontend
        payload = request.get_json()
        print("Received payload:", payload, file=sys.stdout)
        if 'frameData' not in payload:
            raise ValueError('Invalid request payload')

        frame_data = payload['frameData']
        # Decode base64 image data
        image_data = base64.b64decode(frame_data.split(",")[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Process frame using the model
        detected_objects = detect_objects(frame)
         # Log the shape of the received frame
        logging.debug("Received frame shape: %s", frame.shape)
        # Return the detected objects as JSON response
        return jsonify({'detected_objects': detected_objects}), 200
    except Exception as e:
        # Handle any errors gracefully
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
