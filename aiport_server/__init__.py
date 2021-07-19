from flask import Flask, render_template
from flask_cors import CORS

def create_app(test_config=None):
    from aiport_server.main import main_router
    from aiport_server.mask_detector.router import mask_router
    from aiport_server.trip_advisor.router import trip_router

    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(main_router, url_prefix='/main')
    app.register_blueprint(mask_router, url_prefix='/mask-detector')
    app.register_blueprint(trip_router, url_prefix='/trip-advisor')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5001, host='0.0.0.0')
    # 서버 성능이 좋아지면 멀티 코어 설정을 여기서 하면 된다.
    # 사실 제대로 된 gunicorn 사용법 부터 배워야 할 듯.
