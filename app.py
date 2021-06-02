from flask import Flask, render_template

from mask_detector.router import mask_router

app = Flask(__name__)
app.register_blueprint(mask_router, url_prefix='/mask-detector')


@app.route('/')
def router_main():
    return render_template('index.html', 
        info={
            'Name': __name__,
            'Methods': ['GET']
        }
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
