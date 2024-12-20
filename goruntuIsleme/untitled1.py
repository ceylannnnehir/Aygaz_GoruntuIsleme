import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Veri Setinin Hazırlanması
DATASET_DIR = "/kaggle/input/animals-with-attributes/aygaz"  # Veri seti ana klasörü
TARGET_CLASSES = ['antelope', 'bat', 'beaver', 'blue+whale', 'bobcat', 'buffalo', 
                  'chihuahua', 'chimpanzee', 'collie', 'cow']
TARGET_DIR = "processed_dataset"
IMAGE_SIZE = (128, 128)  # Model girişi için kullanılacak boyut

# Hedef dosyaları oluştur
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

for class_name in TARGET_CLASSES:
    source_class_path = os.path.join(DATASET_DIR, class_name)
    target_class_path = os.path.join(TARGET_DIR, class_name)

    if not os.path.exists(target_class_path):
        os.makedirs(target_class_path)

    # İlk 650 resmi taşı
    images = sorted(os.listdir(source_class_path))[:650]
    for img in images:
        shutil.copy2(os.path.join(source_class_path, img), os.path.join(target_class_path, img))

# 2. Verilerin Ön İşlenmesi
X = []
y = []

for class_idx, class_name in enumerate(TARGET_CLASSES):
    class_path = os.path.join(TARGET_DIR, class_name)
    images = os.listdir(class_path)

    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, IMAGE_SIZE)
        img_normalized = img_resized / 255.0
        
        X.append(img_normalized)
        y.append(class_idx)

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(TARGET_CLASSES))

# Eğitim ve Test Seti Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veri Artırma
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 3. CNN Modelinin Tasarlanması
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(TARGET_CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Özeti
model.summary()

# Modelin Eğitilmesi
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=100
)

# 4. Modelin Test Edilmesi
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 5. Manipüle Edilmiş Resim Fonksiyonu
MANIPULATED_DIR = "manipulated_images"
if not os.path.exists(MANIPULATED_DIR):
    os.makedirs(MANIPULATED_DIR)

def get_manipulated_images(input_images, output_dir, brightness_factors=[0.5, 1.0, 1.5]):
    """
    Resimleri farklı parlaklık faktörleri ile manipüle eder ve kaydeder.

    :param input_images: Test resimlerinin dizisi.
    :param output_dir: Manipüle edilmiş resimlerin kaydedileceği klasör.
    :param brightness_factors: Parlaklık faktörlerinin listesi.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, img in enumerate(input_images):
        for factor in brightness_factors:
            manipulated_img = np.clip(img * factor, 0, 1)  # Parlaklığı artır
            manipulated_img = (manipulated_img * 255).astype(np.uint8)  # 0-255 ölçeğine getir

            output_path = os.path.join(output_dir, f"img_{idx}_brightness_{factor}.png")
            cv2.imwrite(output_path, manipulated_img)

# Test resimlerini manipüle et ve kaydet
get_manipulated_images(X_test, MANIPULATED_DIR)

# 6. Manipüle Edilmiş Test Setini Hazırlama

def preprocess_images_for_model(test_images, target_size=(128, 128)):
    processed_images = []
    for img in test_images:
        # Parlaklık manipülasyonu yapılmış görüntülerin boyutlandırılması ve normalizasyonu
        img_resized = cv2.resize(img, target_size)
        img_normalized = img_resized / 255.0
        processed_images.append(img_normalized)
    return np.array(processed_images)

# Manipüle Edilmiş Test Seti Üzerinde Modeli Deneme
X_test_manipulated = preprocess_images_for_model(X_test)  # Manipüle edilmiş test setini hazırlama

# Modelin Manipüle Edilmiş Test Seti ile Test Edilmesi
loss, accuracy = model.evaluate(X_test_manipulated, y_test)
print(f"Manipüle Edilmiş Test Seti ile Test Kayıp: {loss}")
print(f"Manipüle Edilmiş Test Seti ile Test Doğruluğu: {accuracy}")

# 7. Renk Sabitleme Fonksiyonu
def get_wb_images(input_images, output_dir):
    """
    Gray World algoritmasını kullanarak renk sabitleme işlemi yapar.

    :param input_images: Manipüle edilmiş test resimleri.
    :param output_dir: Renk sabitleme sonucu elde edilen görüntülerin kaydedileceği klasör.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, img in enumerate(input_images):
        # Gray World algoritması için kanal ortalamalarını hesapla
        avg_b = np.mean(img[:, :, 0])  # Blue kanal ortalaması
        avg_g = np.mean(img[:, :, 1])  # Green kanal ortalaması
        avg_r = np.mean(img[:, :, 2])  # Red kanal ortalaması

        # Ortalamayı bulduktan sonra, tüm görüntüyeki pikselleri normalize et
        avg = (avg_b + avg_g + avg_r) / 3.0  # Ortalamayı hesapla
        img[:, :, 0] = img[:, :, 0] * (avg / avg_b)  # Mavi kanal
        img[:, :, 1] = img[:, :, 1] * (avg / avg_g)  # Yeşil kanal
        img[:, :, 2] = img[:, :, 2] * (avg / avg_r)  # Kırmızı kanal

        # Normalize edilen görüntüyü kaydet
        output_path = os.path.join(output_dir, f"img_{idx}_gray_world.png")
        img_to_save = (np.clip(img, 0, 1) * 255).astype(np.uint8)  # 0-255 aralığına getir
        cv2.imwrite(output_path, img_to_save)

get_wb_images(X_test_manipulated, "wb_manipulated_images")

# Renk sabitleme uygulanmış test setini yükle
X_test_wb = preprocess_images_for_model(X_test_manipulated)  # Test setini hazırlama

# Modeli renk sabitlemeli test setiyle değerlendirme
loss, accuracy = model.evaluate(X_test_wb, y_test)
print(f"Renk Sabitlemeli Test Seti ile Test Kayıp: {loss}")
print(f"Renk Sabitlemeli Test Seti ile Test Doğruluğu: {accuracy}")
