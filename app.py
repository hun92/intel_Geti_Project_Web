from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from collections import defaultdict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 모델 로드 및 클래스 설정
model = load_model(r'C:\Users\isv44\Desktop\jpg\venv\GETI (2)\model\transfer_learning_model.h5')  # 모델 경로
class_names = ['감자칩', '감자튀김', '도넛', '라면', '마카롱', '치즈볼', '치즈스틱', '치킨', '케이크', '탕후루', '피자', '햄버거']
carbohydrate = [47, 20.33, 49.74, 79, 41.11, 12.24, 20.87, 23.36, 28.58, 8.58, 32.45, 23, 5.86, 2, 69.89]
protein = [27, 1.98, 6.56, 10, 31.77, 8.7, 2.11, 2.35, 9.43, 19.19, 4.67, '-', 1.66, 1, 0.46]
province = [27, 8.01, 37.47, 16, 31.61, 9.4, 4.82, 10.76, 11.26, 14.85, 14.26, '-', 5.86, 6, 0.66]
natrium = [902, 171, 525, 1790, 1233, 183, 13, 257, 536, 260, 98, 12, 99, 65, 75]
sugar = [7, 0.33, 4.12, 4, '-', 0.1, 15.57, 10.58, 1.51, 0.2, 29.66, 22, 24.92, 2, 65.16]

dropdown_dict = {
    "햄버거" : [
        '빅맥',
        '상하이 스파이스',
        '맥스파이시',
        '베이컨 토마토',
        '1995',
        '불고기',
        '맥모닝',
        '더블쿼터파운드치즈버거',
        '치즈버거',
        '더블불고기버거',
        '슈비버거'
        ],
    "감자칩" : [
        '포카칩 오리지널',
        '프링글스 오리지널',
        '프링글스 사워어니언',
        '스윙칩 볶음고추장',
        '예감',
        '눈을감자',
        '허니버터칩',
        '수미칩',
        '오감자',
        '칩포테토'
        ],
    "감자튀김" : [
        '롯데리아 감튀',
        '버거킹 감튀',
        '맥도날드 감튀',
        '맘스터치 감튀'
        ],
    "치즈스틱" : [
        '노브랜드 치즈스틱',
        '오뚜기 치즈스틱',
        '롯데리아 치즈스틱',
        '롯데리아 롱스틱',
        '하림 치즈스틱',
        '맘스터치 치즈스틱',
        '도미노 치즈스틱',
        '맥도날드 치즈스틱'
        ],
    "라면" : [
        '신라면',
        '진라면',
        '열라면',
        '너구리',
        '육개장라면',
        '김치라면',
        '사리곰탕',
        '무파마',
        '오징어짬뽕',
        '참깨라면'
        ],
    "피자" : [
        '포테이토 피자',
        '페퍼로니 피자',
        '슈퍼슈프림',
        '블랙타이거 슈림프',
        '불고기피자',
        '포테이토피자',
        '블랙앵거스 스테이크',
        '슈퍼 디럭스',
        '직화 스테이크',
        '치즈케이크샌',
        '포테이토 씬',
        '슈퍼디럭스 오리지널',
        '리얼바비큐'
        ],
    "치킨" : [
        'BBQ 황금올리브',
        'BBQ 황올순살',
        'BBQ 양념치킨',
        'BHC 뿌링클',
        'BHC 후라이드',
        'BHC 맛초킹',
        '교촌 오리지날',
        '교촌 허니콤보',
        '교촌 레드콤보',
        '굽네 오리지날',
        '굽네 고추바사삭',
        '굽네 볼케이노'
        ],
    "도넛" : [
        '글레이즈드',
        '스트로베리필드',
        '올리브 츄이스티',
        '초코글레이즈드',
        '카카오 허니딥 먼치킨 4조각(52g)',
        '바바리안필드',
        '보스톤크림',
        '스트로베리 먼치킨',
        '카카오하니딥',
        '올드훼션드 먼치킨',
        '카푸치노츄이스티',
        '바바리안 먼치킨 5개 (50g)',
        '소금 우유도넛',
        '초코칩스콘',
        '허니 후리터',
        '페이머스 글레이즈드'
        ],
    
}

calories = [583, 156, 160, 500, 568, 267, 200, 225, 2100, 230, 92, 174, 265, 229, 131]


@app.route('/update_calories', methods=['POST'])
def update_calories():
    if request.method == 'POST':
        selected_calories = request.form.get('calories')
        result = {'status': 'success', 'selected_calories': selected_calories}
        return jsonify(result)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


def process_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        return predicted_class, predictions
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

@app.route('/fileUpload', methods=['POST'])
def file_upload():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename != '':
            current_files = os.listdir(app.config['UPLOAD_FOLDER'])
            for file in current_files:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(request.referrer)  # 파일 업로드 후 이전 페이지로 리다이렉트
    return 'upload failed'

@app.route('/input', methods=['POST'])
def save_weight():
    if request.method == 'POST':
        weight = request.form['user_weight']
        return redirect(url_for('view', weight=weight))  # 입력된 체중 값을 view로 전달합니다.
    return 'Failed to save weight'


@app.route('/view')
def view():
    selected_food = request.args.get('selected_food', None)
    image_list = os.listdir(app.config['UPLOAD_FOLDER'])
    class_counts = defaultdict(int)
    class_frequency = []  # 클래스 빈도수 및 파일 이름을 저장할 리스트
    
    # 체중 값을 가져옵니다. 숫자로 변환할 수 없는 경우 기본값을 설정합니다.
    weight = request.args.get('weight', 75)  # 기본값은 75로 설정합니다.
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        weight = 75  # 기본값 설정

    dropdown_list = [
        '빅맥',
        '상하이 스파이스',
        '맥스파이시',
        '베이컨 토마토',
        '1995',
        '불고기',
        '맥모닝',
        '더블쿼터파운드치즈버거',
        '치즈버거',
        '더블불고기버거',
        '슈비버거',
        '포카칩 오리지널',
        '프링글스 오리지널',
        '프링글스 사워어니언',
        '스윙칩 볶음고추장',
        '예감',
        '눈을감자',
        '허니버터칩',
        '수미칩',
        '오감자',
        '칩포테토',
        '롯데리아 감튀',
        '버거킹 감튀',
        '맥도날드 감튀',
        '맘스터치 감튀',
        '노브랜드 치즈스틱',
        '오뚜기 치즈스틱',
        '롯데리아 치즈스틱',
        '롯데리아 롱스틱',
        '하림',
        '맘스터치 치즈스틱',
        '도미노 치즈스틱',
        '맥도날드',
        '신라면',
        '진라면',
        '열라면',
        '너구리',
        '육개장라면',
        '김치라면',
        '사리곰탕',
        '무파마',
        '오징어짬뽕',
        '참깨라면',
        '포테이토 피자',
        '페퍼로니 피자',
        '슈퍼슈프림',
        '블랙타이거 슈림프',
        '불고기피자',
        '포테이토피자',
        '블랙앵거스 스테이크',
        '슈퍼 디럭스',
        '직화 스테이크',
        '치즈케이크샌',
        '포테이토 씬',
        '슈퍼디럭스 오리지널',
        '리얼바비큐',
        'BBQ 황금올리브',
        'BBQ 황올순살',
        'BBQ 양념치킨',
        'BHC 뿌링클',
        'BHC 후라이드',
        'BHC 맛초킹',
        '교촌 오리지날',
        '교촌 허니콤보',
        '교촌 레드콤보',
        '굽네 오리지날',
        '굽네 고추바사삭',
        '굽네 볼케이노',
        '글레이즈드',
        '스트로베리필드',
        '올리브 츄이스티',
        '초코글레이즈드',
        '카카오 허니딥 먼치킨 4조각(52g)',
        '바바리안필드',
        '보스톤크림',
        '스트로베리 먼치킨',
        '카카오하니딥',
        '올드훼션드 먼치킨',
        '카푸치노츄이스티',
        '바바리안 먼치킨 5개 (50g)',
        '소금 우유도넛',
        '초코칩스콘',
        '허니 후리터',
        '페이머스 글레이즈드']

    # 이미지 예측 및 빈도수 계산
    for filename in image_list:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            
            dropdown_list = dropdown_dict.get(predicted_class, [])

            class_counts[predicted_class] += 1
            # 'calories' 키를 딕셔너리에 추가합니다.
            # 실제 칼로리 값은 여기에 추가해주셔야 합니다.
            class_frequency.append({
                'class_name': predicted_class,
                'filename': filename,
                'calories': calories[np.argmax(predictions)],
                'carbohydrate': carbohydrate[np.argmax(predictions)],
                'protein': protein[np.argmax(predictions)],
                'province': province[np.argmax(predictions)],
                'natrium': natrium[np.argmax(predictions)],
                'sugar': sugar[np.argmax(predictions)]
                # 다른 칼로리 관련 정보도 동일하게 추가하셔야 합니다.
            })

            class_counts[predicted_class] += 1

        except (IndexError, FileNotFoundError) as e:
            # 오류 발생 시 이를 출력하여 디버깅하거나 로깅합니다.
            print(f"Error processing {filename}: {e}")
            # 이 오류를 무시하고 계속 진행할 것인지 아니면 중단할 것인지 결정합니다.
            pass
    return render_template('view.html', class_frequency=class_frequency, weight=weight, dropdownList=dropdown_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
