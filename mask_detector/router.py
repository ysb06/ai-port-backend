from flask import request, Blueprint, render_template, Response
from flask.helpers import make_response
import matplotlib.image as img
import matplotlib.pyplot as plt

mask_router = Blueprint('mask-detector', __name__)

@mask_router.route('/', methods=('GET', 'POST', 'OPTIONS'))
def router_main():
    if request.method == "GET":    
        return render_template('index.html', 
            info={
                'Name': __name__
            },
            content='Content'
        )
    elif request.method == 'POST':
        target = request.files['imageFile']
        # target을 파일처럼 읽으면 된다.
        return Response('OK')
    else:
        res = Response()
        res.headers.add("Access-Control-Allow-Origin", "*")
        res.headers.add('Access-Control-Allow-Headers', "*")
        res.headers.add('Access-Control-Allow-Methods', "GET,DELETE")
        return res