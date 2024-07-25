import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# データセットのパス
train_dir = "train"
validation_dir = "validation"

# 画像サイズとバッチサイズの定義
img_height, img_width = 224, 224  # 画像サイズは適宜変更
batch_size = 32

# トレーニングデータセットの読み込み
train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

# 検証データセットの読み込み
validation_dataset = image_dataset_from_directory(
    validation_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

# モデルの定義
model = Sequential([
    #畳み込み層＋プーリング層
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    #畳み込み層＋プーリング層
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    #畳み込み層＋プーリング層
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    #フラットニング層 + 全結合層 + ドロップアウト層 + 出力層
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')  
])

# モデルのコンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# モデルの学習
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30  
)

# モデルの評価
loss, accuracy = model.evaluate(validation_dataset)
print(f"Validation loss: {loss}")
print(f"Validation accuracy: {accuracy}")

# トレーニング結果の可視化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.savefig('result.png')

# モデルの保存
model.save('pokemon_classification_model_newdata.h5')
