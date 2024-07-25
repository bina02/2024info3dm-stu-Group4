import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def predict_machine(path):
    # 저장된 모델 로드
    #model_path = 'folders/VGGー16＋alpha/vgg16_fine_trained_model.h5' #first model
    model_path = 'models/vgg16_finetrained_epoch=16.h5' #second model

    loaded_model = load_model(model_path)

    # 예측할 이미지 경로
    image_path = path

    # 이미지 불러오기 및 전처리
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.  # 이미지 정규화 (학습 시와 동일한 전처리 필요)

    # 예측 수행
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # 클래스 인덱스에 따른 레이블 설정
    class_labels = ['eevee', 'espeon', 'flareon', 'glaceon', 'jolteon', 'leafeon', 'sylveon', 'umbreon', 'vaporeon']  # 클래스 레이블 리스트 설정

    # 예측 결과 반환
    return [class_labels[predicted_class], f'{predictions[0][predicted_class] * 100:.2f}%']
list = []
filePath = 'nuigurumi/IMG_1465.HEIC'
result = predict_machine(filePath)
print([result[0], result[1]])


