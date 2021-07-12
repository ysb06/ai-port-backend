from flask import Blueprint, jsonify, render_template, request

trip_router = Blueprint('trip-advisor', __name__)

@trip_router.route('/', methods=('GET', 'POST'))
def router_main():
    if request.method == "GET":    
        return render_template(
        'index.html', 
        info={
            'Name': __name__,
            'Methods': ['GET', 'POST']
        }
    )
    elif request.method == 'POST':
        result = 'Post test good'
        return jsonify({'result': result})
