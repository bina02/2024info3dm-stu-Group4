import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlowのログを抑制

# 予測関数
def predict_machine(path, model_path):
    # モデルのロード
    loaded_model = load_model(model_path)

    # 画像の読み込みと前処理
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 画像の正規化

    # 予測の実行
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # クラスラベルの設定
    class_labels = ['eevee', 'espeon', 'flareon', 'glaceon', 'jolteon', 'leafeon', 'sylveon', 'umbreon', 'vaporeon']

    # 予測結果の返却
    return class_labels[predicted_class], predictions[0][predicted_class] * 100

# 予測する画像ファイルのパス
img_path = '/Users/hwanghyeonbin/report/report3-2/知能情報総合実験/folders/nuigurumi/IMG_1466.jpeg'  # ここに対象の画像ファイルのパスを指定

# モデルのパスリスト
model_paths = [
    'models/vgg16_finetrained_prototype.h5',
    'folders/models/vgg16_finetrained_epoch=10.h5',
    'folders/models/vgg16_finetrained_epoch=16.h5'
]

# 結果をコンソールに出力
print(f"Image: {os.path.basename(img_path)}\n")

try:
    for model_path in model_paths:
        # モデルの予測を実行
        result_class, result_confidence = predict_machine(img_path, model_path)
        
        # 予測結果をコンソールに出力
        print(f"Using model: {model_path}")
        print(f"  Class - {result_class}, Confidence - {result_confidence:.2f}%\n")
            
except Exception as e:
    print(f"Error processing file {img_path}: {e}")
