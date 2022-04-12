import dash
import dash_core_components as dcc
import dash_html_components as html

from emotion_detection import *

from flask import Flask, Response
import cv2

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        
        return jpeg.tobytes()
    def compute_labels(self):
        (grabbed, frame) = self.video.read()

        # clone the current frame, convert it from BGR into RGB
        frame = utils.resize_image(frame, width=720, height=720)
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # initialize an empty canvas to output the probability distributions
        canvas = np.zeros((300, 300, 3), dtype="uint8")

        # get the frame dimension, resize it and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300))

        # infer the blob through the network to get the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # iterate over the detections
        for i in range(0, detections.shape[2]):

            # grab the confidence associated with the model's prediction
            confidence = detections[0, 0, i, 2]

            # eliminate weak detections, ensuring the confidence is greater
            # than the minimum confidence pre-defined
            if confidence > args['confidence']:

                # compute the (x,y) coordinates (int) of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # grab the region of interest within the image (the face),
                # apply a data transform to fit the exact method our network was trained,
                # add a new dimension (C, H, W) => (N, C, H, W) and send it to the device
                face = frame[start_y:end_y, start_x:end_x]
                face = data_transform(face)
                face = face.unsqueeze(0)
                face = face.to(device)

                # infer the face (roi) into our pretrained model and compute the
                # probability score and class for each face and grab the readable
                # emotion detection
                predictions = model(face)
                prob = nnf.softmax(predictions, dim=1)
                top_p, top_class = prob.topk(1, dim=1)
                top_p, top_class = top_p.item(), top_class.item()

                # grab the list of predictions along with their associated labels
                emotion_prob = [p.item() for p in prob[0]]
                emotion_value = emotion_dict.values()
                # print()
                # draw the probability distribution on an empty canvas initialized
                # for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
                #     prob_text = f"{emotion}: {prob * 100:.2f}%"
                face_emotion = emotion_dict[top_class]
                # face_text = f"{face_emotion}: {top_p * 100:.2f}%"
                return emotion_prob,emotion_value,face_emotion
            


def gen(camera):
    while True:
        frame = camera.get_frame()
        # emotion_prob,emotion_value,face_emotion=camera.compute_labels()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
       
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div(style={'backgroundColor': '#F1F1F1'},children=[
    html.H1("Emotion"),
    html.Img(src="/video_feed"),
    # dcc.Graph(figure=)
])

if __name__ == '__main__':
    app.run_server(debug=True)
