from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Only detect monitors/TVs (class index 63 is TV/monitor for the COCO dataset)
allowed_classes = [63]

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use the device's default camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame)
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if int(cls) in allowed_classes:
                    # Draw the bounding box for monitors/TVs
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Encode the frame into JPEG format to stream in the browser
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
