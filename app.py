from flask import Flask, render_template, Response
import cv2
import pickle
import numpy as np
import mediapipe as mp

app = Flask(__name__)


def capture_by_frames_arsl():
    global camera
    camera = cv2.VideoCapture(0)
    model_dict = pickle.load(open('./model28.1.p', 'rb'))
    model = model_dict['model']
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'Alif', 1: 'Ba', 2: 'Ta', 3: 'Cha', 4: 'Jim', 5: 'Ha', 6: 'Kha', 7: 'Dal', 8: 'D\'hal', 9: 'Ra',
                   10: 'Ja', 11: 'Siin', 12: 'Shiin', 13: 'Sod', 14: 'Dod', 15: 'Toa', 16: 'Joa', 17: 'A\'yn',
                   18: 'Ga\'yn', 19: 'Fa', 20: 'Q\'of', 21: 'Kaf', 22: 'Lam', 23: 'Mim', 24: 'Wao', 25: 'Nun',
                   26: 'Hamja', 27: 'Ya', 28: 'Nothing'}
    while True:
        data_aux = []
        x_ = []
        y_ = []
        # read the camera frame
        success, frame = camera.read()
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/course', methods=['GET'])
def blog():
    return render_template('course.html')


@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')


@app.route('/single', methods=['GET'])
def single():
    return render_template('single.html')


@app.route('/sign_ai', methods=['GET'])
def sign_ai():
    return render_template('sign_ai.html')


@app.route('/arsl', methods=['GET'])
def arsl():
    return render_template('arsl.html')


@app.route('/start_arsl', methods=['POST'])
def start_arsl():
    return render_template('start_arsl.html')


@app.route('/stop_arsl', methods=['POST'])
def stop_arsl():
    if camera.isOpened():
        camera.release()
    return render_template('stop_arsl.html')


@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames_arsl(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
