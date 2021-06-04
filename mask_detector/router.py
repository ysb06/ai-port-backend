import matplotlib.image as img
import matplotlib.pyplot as plt
from flask import Blueprint, Response, render_template, request
from flask.helpers import make_response

from mask_detector.predictor import finalize, initialize, predict

mask_router = Blueprint('mask-detector', __name__)

@mask_router.route('/', methods=('GET', 'POST'))
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

        initialize()
        result = predict(target)
        finalize()

        return Response('OK')
