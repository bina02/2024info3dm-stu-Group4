import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 保存されたモデルを読み込む
model = tf.keras.models.load_model('pokemon_classification_model.h5')

# 予測関数の定義
def predict_image(img_path):
    class_labels = ['eevee', 'espeon', 'flareon', 'glaceon', 'jolteon', 'leafeon', 'umbreon', 'vaporeon', 'sylveon']  # クラスのラベルを定義
    
    # 画像を読み込む
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 予測を行う
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    
    # 結果を出力
    print(f"Predicted class: {predicted_class}")

# 予測の実行例
for i in range(1, 9):
    img_path = f'./img/flareon{i}.jpeg'  # 画像パスを生成
    print(f"Predicting for image: {img_path}")
    predict_image(img_path)
