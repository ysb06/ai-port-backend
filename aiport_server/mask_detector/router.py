from aiport_server.mask_detector.predictor import (model_finalize,
                                                   model_initialize, predict)
from flask import Blueprint, jsonify, render_template, request

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

        model_initialize()
        result = predict(target)
        model_finalize()
        # 원래는 initalize를 서버 시작과 함께 불러서 모델을 RAM에 상주시켜 속도를 높이고 싶었지만
        # GCP f1-micro 서버로는 RAM이 부족해서 다른 모델을 사용하려면 OOM이 발생할 것으로 예측되므로
        # 현재는 요청과 함께 모델을 하드디스크로부터 불러오고 사용 완료되면 메모리에서 삭제하는 식으로 동작

        # print(f"Predict Complete: {result}")
        return jsonify({'result': result})
