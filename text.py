import os
import argparse
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from ollama import generate

# 蔬菜类别列表
DEFAULT_CLASSES = [
    "tomato", "胡萝卜", "黄瓜", "茄子", "土豆",
    "青椒", "西兰花", "生菜", "白菜", "洋葱"
]

def load_model(model_path):
    """加载预训练的Keras模型"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"模型加载成功: {model_path}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """预处理图像用于模型输入"""
    try:
        image = Image.open(image_path).convert("RGB")
        # 添加图片缩放功能
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None

def predict_vegetable(model, image, class_names):
    """预测蔬菜类别"""
    try:
        predictions = model.predict(image)
        prediction_index = np.argmax(predictions[0])
        confidence = round(100 * predictions[0][prediction_index], 2)
        predicted_class = class_names[prediction_index]
        return predicted_class, confidence
    except Exception as e:
        print(f"预测过程出错: {e}")
        return None, None

def generate_vegetable_info(vegetable_name, ollama_model="llama3.2"):
    """使用Ollama API生成蔬菜信息"""
    prompt = f"请详细介绍蔬菜'{vegetable_name}'的营养价值和烹饪方法..."
    response = generate(model=ollama_model, prompt=prompt, max_tokens=1000)
    return response.response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vegetable_recognition_model.h5', help='模型路径')
    parser.add_argument('--classes', default=','.join(DEFAULT_CLASSES), help='类别列表')
    parser.add_argument('--output', help='输出文件路径')
    args = parser.parse_args()

    image_path = '0c77208843c640a3.jpg'
    if not os.path.exists(image_path):
        print(f"错误: 未找到图像文件 '{image_path}'")
        return

    model = load_model(args.model)
    if not model:
        return

    # 直接使用模型期望的尺寸预处理图像
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return

    class_names = args.classes.split(',')
    predicted_class, confidence = predict_vegetable(model, processed_image, class_names)

    if predicted_class and confidence:
        print(f"识别结果: {predicted_class} (置信度: {confidence}%)")
        # 生成信息并输出...

if __name__ == "__main__":
    main()