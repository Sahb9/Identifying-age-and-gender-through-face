# %% [0] Import libraries and modules
from statistics import mode
import joblib
from keras.utils import plot_model
from PIL import Image
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from keras.models import Sequential, Model
#from keras.preprocessing.image import load_img
from sklearn.metrics import mean_squared_error
from keras_preprocessing.image import load_img
from sklearn.metrics import accuracy_score
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from tensorflow import keras

warnings.filterwarnings('ignore')


# %% Load data

DATASET_PATH = 'UTKFace'

# Gắn nhãn tuổi và giới tính vào biến
image_paths = []
age_labels = []
gender_labels = []

for filename in os.listdir(DATASET_PATH):
    image_path = os.path.join(DATASET_PATH, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

# %% Chuyển dữ liệu trên thành 1 data frame
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head()

# định nghĩa về giới tính theo số
gender_dict = {0: 'Male', 1: 'Female'}

# %% Phân tích dữ liệu
# Hiển thị hình mẫu
img = Image.open(df['image'][0])
plt.axis('off')
plt.imshow(img)

#%% Hiển thị độ phân phối độ tuổi của tập dữ liệu
sns.distplot(df['age'])
#%% Hiển thị độ phân phối giới tính của tập dữ liệu
sns.countplot(df['gender'])

#%% Hiển thị các hình mẫu cùng với các label
plt.figure(figsize=(20, 20))
files = df.iloc[0:25]

for index, file, age, gender in files.itertuples():
    plt.subplot(5, 5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Age: {age} Gender: {gender_dict[gender]}")
    plt.axis('off')

# %% Phân tách feature
# Hàm sử dụng để phân tách hình ảnh thành các feature
# Có 128 feature tương ứng với độ phân giải 128x128
def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, grayscale=True)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features


X = extract_features(df['image'])
print(X.shape)
# Chuẩn hóa dữ liệu hình ảnh về dạng khoảng [0,1]
X = X/255.0

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])


# %% Tạo model
input_shape = (128, 128, 1)
inputs = Input((input_shape))
# Convolutional layers (Lớp Tích Chập)
# Cùng với Pooling layers (Lớp Tổng Hợp)
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

# Fully connected layers (Lớp kết nối đầy đủ)
dense_1 = Dense(256, activation='relu')(flatten)
dense_2 = Dense(256, activation='relu')(flatten)

dropout_1 = Dropout(0.3)(dense_1)
dropout_2 = Dropout(0.3)(dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
output_2 = Dense(1, activation='relu', name='age_out')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'],
              optimizer='adam', metrics=['accuracy'])

model.summary

# Split model
X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
    X, y_gender, y_age, test_size=0.10, random_state=42)

#%% train model
my_callbacks = [
    # Dừng sớm khi sau 3 epoch mà không có sự cải thiện của validation loss
    tf.keras.callbacks.EarlyStopping(patience = 3, monitor = 'val_loss'),
    # Nếu sau epoch không có sự cải thiện thì giảm LR
    tf.keras.callbacks.ReduceLROnPlateau(patience = 2)
]

run = 1
if run == 1:
    history = model.fit(
        x=X_train, y=[y_gender_train, y_age_train], batch_size=32,
         epochs=30, callbacks = my_callbacks, validation_split=0.2)
    model.save("model_2/NhanDangTuoi_GioiTinh_model")
else:
    model = keras.models.load_model("model_2/NhanDangTuoi_GioiTinh_model.h5")

# %% Hiển thị kết quả cho giới tính
acc = history.history['gender_out_accuracy']
val_acc = history.history['val_gender_out_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['gender_out_loss']
val_loss = history.history['val_gender_out_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

#np.save('model_2/my_history.npy',history.history)
#history=np.load('model_2/my_history.npy',allow_pickle='TRUE').item()

#%%
if run == 1:
    predict_model_test = model.predict(X_test)
    joblib.dump(predict_model_test, r'models/predict_model_test.pkl')
else:
    predict_model_test = joblib.load('models/predict_model_test.pkl')


test_acc = accuracy_score(np.array(y_gender_test), np.around(predict_model_test[0]))

print('Độ chính xác trên tập train: ')
print(acc[29])
print('Độ chính xác trên tập validation: ')
print(val_acc[29])
print('Độ chính xác trên tập test: ')
print(test_acc)

# %% Hiển thị kết quả về tuổi
loss = history.history['age_out_loss']
val_loss = history.history['val_age_out_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

#%%
print("Loss trên tập train : ",loss[29])
print("Loss trên tập validation :",val_loss[29])

mse_test = mean_squared_error(np.array(y_age_test).astype('float32').reshape((-1,1)),
        np.around(predict_model_test[1]))

RMSE_test=np.sqrt(mse_test)
print("Loss trên tập test :",RMSE_test)

test_acc = accuracy_score(np.array(y_age_test), np.around(predict_model_test[1]))

print('Độ chính xác trên tập train: ')
print(acc)
print('Độ chính xác trên tập test: ')
print(val_acc)


#%% Dự đoán với dữ liệu test

#80
image_index = 80
print("Original Gender:",
      gender_dict[y_gender_test[image_index]], "Original Age:", y_age_test[image_index])
# predict from model
pred = model.predict(X_test[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X_test[image_index].reshape(128, 128), cmap='gray')

# %%
