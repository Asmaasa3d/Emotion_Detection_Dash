from asyncio import events
from multiprocessing import Event
from re import X
from turtle import color
from cv2 import add
import dash
# from hamcrest import none
import plotly
# import dash_core_components as dcc
# import dash_html_components as html
from typing import Type
from dash.dependencies import Input,Output,State
# from matplotlib.figure import Figure
from collections import deque
import pandas as pd
import plotly.express as px
from plotly import graph_objs as go
from dash import html,callback_context
from emotion_detection import *
import dash_bootstrap_components as dbc
from dash import dcc
from flask import Flask, Response
import cv2

X = deque(maxlen = 1000)
X.append(1)

Y = deque(maxlen = 1000)
Y.append(1)

moods_list = {
    'Happy': '😂',
    'Content': '🙂',
    'Neutral': '😐',
    'Sad': '☹️',
    'Angry': '😡',
    'Bored': '😒',
    'Tired': '😫',
    'Grateful': '😇',
    'Stressed': '😓',
    'Motivated': '🧐',
    'Relieved': '😌',
    'fFocused': '🤔',
    'Irritated': '😩',
    'Relaxed': '😎',
    'Hopeful': '😏',
    'Fearful': '😰',
    'Frustrated': '😖',
    'Inspired': '🤩',
    'Guilt': '🤥',
    'Ashamed': '😬',
    'Depressed': '😥',
    'Indifferent': '😕',
    'Surprised' :'😱'
}


class VideoCamera(object):
    def __init__(self,mode):
        # cv2.destroyAllWindows()
        if mode==-1:
            self.video = cv2.VideoCapture(0)
        elif mode==0:
            self.video = cv2.VideoCapture('video/Facebook media.mp4')

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        success, image = self.video.read()
        if  success:
            
            ret, jpeg = cv2.imencode('.jpg', image)
        
            return jpeg.tobytes()
        else:
            return 0
            
    def compute_labels(self):
        (grabbed, frame) = self.video.read()
        if not grabbed:
            return
        
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
        # print( detections.shape[2])
        for i in range(0, detections.shape[2]):
        # i=0
        # grab the confidence associated with the model's prediction
            confidence = detections[0, 0, i, 2]

            # eliminate weak detections, ensuring the confidence is greater
            # than the minimum confidence pre-defined
            # if confidence > args['confidence']:

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
            # print(emotion_value, face_emotion, emotion_prob)
            # print(face_emotion)
            data = {'mood':["Angry", "Fearful", "Happy", "Neutral",
                    "Sad","Surprised"],
        'prob':emotion_prob}
            return data
            # return emotion_value, face_emotion, emotion_prob
            
Data = {'mood':["Angry", "Fearful", "Happy", "Neutral",
                    "Sad","Surprised"],
        'prob':[0, 0, 0,0, 0, 0]}
DataAll = pd.DataFrame(Data,columns=['mood', 'prob'])
def gen(camera):
    
    while True:
        frame = camera.get_frame()
        if frame==0:
            break
        global data 
        data = camera.compute_labels() 
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        # yield from((b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'),data
        # )

# SUPERHERO VAPOR CERULEAN LUX MATERIA 

server = Flask(__name__)
app = dash.Dash(__name__, server=server,external_stylesheets=[dbc.themes.FLATLY,'./assets/Stylesheet.css','./assets/style.scss','//cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'],prevent_initial_callbacks=True)

# app.config["suppress_callback_exceptions"] = True
app.css.config.serve_locally = True

@server.route('/video_feed1')
def video_feed1():    
    return Response(gen(VideoCamera(-1)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@server.route('/video_feed2')
def video_feed2():    
    return Response(gen(VideoCamera(0)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
app.layout = html.Div(
    [
# First Row 
            # ------------------------------------------ Header Section 1  ------------------------------------------ #
        dbc.Row(dbc.Col(html.Div([html.H1(" EM🤩TION  B😒ARD  "), html.Hr(className="separator separator--dots")]
                                ,style = {'padding-bottom' : '5%','text-align': 'center'}))
        ),

# scond Row   #===========##===========##===========##===========##===========#
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Div(dbc.CardHeader(' Facial emotion tracker ',style={'text-align': 'center'},className="me-1"),className='card-header p-3 mb-2 bg-secondary text-white'),
                                html.Div( html.Img(src='/video_feed2',id='emotion',height=500,width=550,className='center')),
                        
                                html.Div(
                                    [
                                        dbc.Button(
                                                html.Span(["Video   " ,html.I(className="fa fa-youtube-play", n_clicks=0,style={'font-size': '30px','verticalAlign': 'centre','color':'Tomato'})],style={'font-size': '30px'})
                                                ,id='btn-nclicks-1',style={"border-radius": "5px",'margin-right': '15px'}, outline=True,className="me-1",color="primary"
                                                    ),
                                        dbc.Button(
                                                html.Span(["Camera  ",html.I(className="fa fa-video-camera", n_clicks=0,style={'font-size': '35px','verticalAlign': 'centre','color':'DodgerBlue'})],style={'font-size': '30px'})
                                                            ,id='btn-nclicks-2',style={'margin-right': '15px', "border-radius": "5px"}, outline=True,className="me-1",color="primary"
                                                    ),
                                        dbc.Button("Start", color="primary",n_clicks=0, className="me-1",id='start-id',outline=True, style={'margin-left': '200px'}),
                                    ], style = {'padding-left' : '10%','padding-top' : '8%'}
                                        )
                            ]
                            ,className='cardl',style={'max-width':'730px'})
                        
                        ],style = {'padding-left' : '2%'}),
                
                dbc.Col(
                    [  
                        html.Div(
                            [
                                html.Div(
                                    [
                                    html.Div(dbc.CardHeader('Percentage Emotion',style={'text-align': 'center'}),className='card-header p-3 mb-2 bg-secondary text-white'),
                                    dcc.Graph(id = 'live-graph',className="card-body"), #,animate = True
                                    dcc.Interval(
                                        id = 'graph-update',
                                        interval = 1000,
                                        n_intervals = 0
                                        )
                                    ] ,className="card border-primary mb-3",style={'max-hight':'300px'}),
                                html.Div(
                                    [
                                        html.Div(dbc.CardHeader(' Emotion over time',style={'text-align': 'center'}),className='card-header p-3 mb-2 bg-secondary text-white'),
                                        dcc.Graph(id = 'line-graph' , className="card-body")
                                    ],className="card border-primary mb-3",style={'max-hight':'300px'})
                            ] , className='cardl'
                                )
                    ]
                        )
            ]
            ,style = {'padding-right' : '2%','padding-left' : '2%','padding-bottom' : '15%'}
    ),

# Third Row   #===========##===========##===========##===========##===========#
    dbc.Row(
        [
            dbc.Col(html.A(html.I(className="fa fa-linkedin-square", style={'font-size':'48px','color':'DodgerBlue'}),
                        href='https://github.com/czbiohub/singlecell-dash/issues/new'), width="auto"),
            dbc.Col(html.A(html.I(className="fa fa-github", style={'font-size':'48px','color':'#000000'}),
                        href='https://github.com/czbiohub/singlecell-dash/issues/new'), width="auto"),],
        justify="center",
    )
    
    ]
)


@app.callback(
    [Output('live-graph', 'figure')],
    Input('start-id','n_clicks'),
    Input('graph-update', 'n_intervals')
)
def update_graph_bar(n_clicks,n_interva):
            colors = {'A': 'DodgerBlue',
            'B': 'red'}
    
            global data
            df = pd.DataFrame(data,columns=['mood', 'prob'])
            # print
            df['emojis'] = df['mood'].map(moods_list)
            fig =  plotly.graph_objs.Bar(x=df['mood'],y =df[ 'prob'],text =df['emojis'],textfont_size=14, textangle=0, textposition="outside", cliponaxis=False ,marker={'color': colors['A']})
            
            return [go.Figure(data=fig).update_layout(
                                                    plot_bgcolor='rgb(255, 255, 255)'
                                                    ,xaxis=dict(showgrid=False),
                        yaxis={'title': 'x-label',
                                'visible': False,
                                'showticklabels': False},height=270,margin=dict(
                # pad=5,
                r = 10,
                l=10,
                b=10,
                t=30
                                )
                    )
        ]

@app.callback(
    [Output('line-graph', 'figure')],
    Input('graph-update', 'n_intervals')
)

def update_graph_line(n):
    global data
    global DataAll
    df = pd.DataFrame(data,columns=['mood', 'prob'])
    DataAll['emojis'] = DataAll['mood'].map(moods_list)
    df['emojis'] = df['mood'].map(moods_list)
    DataAll  = DataAll.append(df,ignore_index=True)
    DataAll.append(df)
    DataAll[ '60s_rolling_avg' ] = DataAll['prob'].rolling(1000).mean()
    # print(DataAll)
    # X.append(X[-1]+1)

    DataAll['emojis'] = DataAll['mood'].map(moods_list)
    fig = px.line(DataAll, y='prob', color='emojis', markers=True, height=300)
    # fig =plotly.graph_objs.Scatter(x=X, y =df[ 'prob'],text =df['emojis'],mode='lines+markers',marker=dict(size=[4, 6, 8, 10],
    #             color=[0, 1, 2, 3]),textfont_size=14, cliponaxis=False )

    return [go.Figure(data=fig).update_layout(plot_bgcolor='rgb(255, 255, 255)',
                                        xaxis={'visible': False,
                        'showticklabels': False},
                yaxis={
                        'visible': False,
                        'showticklabels': True},height=270,margin=dict(
        # pad=5,
        l=10,
        b=10,
        t=20
                        ))
]


@app.callback(
    Output('emotion', 'src'),
    Input('btn-nclicks-1', 'n_clicks_timestamp'),
    Input('btn-nclicks-2', 'n_clicks_timestamp')
    # [State('emotion', 'src')
    
)
def displayClick(btn1, btn2):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        msg = '/video_feed2'
    elif 'btn-nclicks-2' in changed_id:
        msg = '/video_feed1'
    # else:
    #     msg = 'None of the buttons have been clicked yet'
    return msg   

if __name__ == '__main__':
    app.run_server(debug=True)


