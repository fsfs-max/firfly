import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 页面配置
st.set_page_config(
    page_title="蔬菜识别助手",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 标题和介绍
st.title("蔬菜识别助手")
st.markdown("上传一张蔬菜图片，我会帮你识别它是什么蔬菜！本应用支持识别以下常见蔬菜：")

# 蔬菜类别列表
vegetable_classes = [
    "白菜", "菠菜", "西兰花", "胡萝卜", "黄瓜", "茄子", "青椒", "土豆", "西红柿", "洋葱"
]

# 显示支持的蔬菜样本
st.subheader("支持识别的蔬菜样本")
cols = st.columns(5)
for i, veg in enumerate(vegetable_classes[:5]):
    with cols[i]:
        st.image(f"https://picsum.photos/seed/{veg}1/200/200", caption=veg, use_column_width=True)

cols = st.columns(5)
for i, veg in enumerate(vegetable_classes[5:]):
    with cols[i]:
        st.image(f"https://picsum.photos/seed/{veg}2/200/200", caption=veg, use_column_width=True)


# 模型定义
class VegetableClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(VegetableClassifier, self).__init__()
        # 使用预训练的ResNet18模型
        self.model = models.resnet18(pretrained=True)

        # 冻结大部分预训练层
        for param in list(self.model.parameters())[:-5]:
            param.requires_grad = False

        # 修改最后的全连接层以适应我们的分类任务
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# 加载模型
@st.cache_resource
def load_model():
    model = VegetableClassifier(len(vegetable_classes))
    # 这里应该加载实际的模型权重
    # 为了演示，我们创建一个随机初始化的模型
    # 在实际应用中，你需要训练模型并加载权重
    # model.load_state_dict(torch.load('vegetable_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


# 图像预处理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# 预测函数
def predict(image, model):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = probs.topk(5, dim=1)

    results = []
    for i in range(top_class.size(1)):
        results.append({
            'class': vegetable_classes[top_class[0, i].item()],
            'probability': top_prob[0, i].item() * 100
        })

    return results


# 主界面
st.subheader("上传蔬菜图片进行识别")
uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

# 示例图片选择器
st.subheader("或者从示例图片中选择")
example_images = {
    "白菜": "https://picsum.photos/seed/cabbage/400/300",
    "菠菜": "https://picsum.photos/seed/spinach/400/300",
    "西兰花": "https://picsum.photos/seed/broccoli/400/300",
    "胡萝卜": "https://picsum.photos/seed/carrot/400/300",
    "黄瓜": "https://picsum.photos/seed/cucumber/400/300"
}

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("白菜"):
        response = requests.get(example_images["白菜"])
        uploaded_file = BytesIO(response.content)
with col2:
    if st.button("菠菜"):
        response = requests.get(example_images["菠菜"])
        uploaded_file = BytesIO(response.content)
with col3:
    if st.button("西兰花"):
        response = requests.get(example_images["西兰花"])
        uploaded_file = BytesIO(response.content)
with col4:
    if st.button("胡萝卜"):
        response = requests.get(example_images["胡萝卜"])
        uploaded_file = BytesIO(response.content)
with col5:
    if st.button("黄瓜"):
        response = requests.get(example_images["黄瓜"])
        uploaded_file = BytesIO(response.content)

# 模型加载状态
with st.spinner("正在加载模型..."):
    model = load_model()

# 处理上传的图片
if uploaded_file is not None:
    try:
        # 读取图片
        if isinstance(uploaded_file, BytesIO):
            image = Image.open(uploaded_file).convert('RGB')
        else:
            image = Image.open(uploaded_file).convert('RGB')

        # 显示上传的图片
        st.image(image, caption='上传的图片', use_column_width=True)

        # 预测
        with st.spinner("正在识别..."):
            results = predict(image, model)

        # 显示结果
        st.subheader("识别结果")
        for i, result in enumerate(results):
            confidence_color = "green" if result['probability'] > 70 else "orange" if result[
                                                                                          'probability'] > 30 else "red"
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="color: {confidence_color};">{i + 1}. {result['class']} (置信度: {result['probability']:.2f}%)</h4>
                <div class="progress" style="height: 25px; background-color: #e9ecef; border-radius: 5px;">
                    <div class="progress-bar" role="progressbar" style="width: {result['probability']}%; background-color: {confidence_color}; color: white; font-weight: bold;" aria-valuenow="{result['probability']}" aria-valuemin="0" aria-valuemax="100">
                        {result['probability']:.2f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 显示蔬菜信息
        if results[0]['probability'] > 50:  # 只有置信度足够高时才显示
            selected_veg = results[0]['class']
            st.subheader(f"关于 {selected_veg} 的信息")

            # 蔬菜信息数据库
            veg_info = {
                "白菜": "白菜是十字花科蔬菜，富含维生素C和纤维素，适合炒、炖和腌制。",
                "菠菜": "菠菜富含铁和维生素K，是补血佳品，适合清炒或做汤。",
                "西兰花": "西兰花是营养丰富的十字花科蔬菜，富含维生素C和叶酸，适合清炒或凉拌。",
                "胡萝卜": "胡萝卜富含β-胡萝卜素，对眼睛有益，适合炒、炖或生食。",
                "黄瓜": "黄瓜水分含量高，适合生食、凉拌或做沙拉，也可用于美容。",
                "茄子": "茄子是茄科蔬菜，富含维生素P，适合烧、炖或烤。",
                "青椒": "青椒富含维生素C和抗氧化物质，适合炒肉或做配菜。",
                "土豆": "土豆是全球第四大粮食作物，富含碳水化合物，适合煮、炸、烤等多种烹饪方式。",
                "西红柿": "西红柿富含番茄红素，是一种抗氧化剂，适合炒、煮汤或生食。",
                "洋葱": "洋葱含有前列腺素A，能降低外周血管阻力，适合炒、烤或生食。"
            }

            st.info(veg_info.get(selected_veg, "抱歉，暂无该蔬菜的信息。"))

            # 显示烹饪建议
            cooking_tips = {
                "白菜": "白菜炒豆腐、酸辣白菜、白菜炖粉条",
                "菠菜": "清炒菠菜、菠菜蛋花汤、菠菜拌粉丝",
                "西兰花": "蒜蓉西兰花、西兰花炒虾仁、白灼西兰花",
                "胡萝卜": "胡萝卜炖排骨、胡萝卜炒肉丝、胡萝卜鸡蛋饼",
                "黄瓜": "拍黄瓜、黄瓜炒鸡蛋、黄瓜沙拉",
                "茄子": "鱼香茄子、红烧茄子、地三鲜",
                "青椒": "青椒炒肉丝、虎皮青椒、青椒土豆丝",
                "土豆": "酸辣土豆丝、土豆烧牛肉、炸薯条",
                "西红柿": "西红柿炒鸡蛋、西红柿鸡蛋汤、糖拌西红柿",
                "洋葱": "洋葱炒牛肉、洋葱圈、凉拌洋葱"
            }

            st.markdown(f"**烹饪建议**：{cooking_tips.get(selected_veg, '暂无烹饪建议')}")

    except Exception as e:
        st.error(f"处理图片时出错: {str(e)}")
        st.exception(e)

# 关于页面
with st.sidebar:
    st.title("关于蔬菜识别助手")
    st.markdown("""
    本应用使用深度学习技术识别常见蔬菜。
    它可以帮助用户快速识别蔬菜种类，并提供相关的营养信息和烹饪建议。

    ### 技术细节
    - 使用PyTorch训练的ResNet18模型
    - 支持识别10种常见蔬菜
    - 部署在Streamlit Cloud平台上

    ### 注意事项
    - 本应用仅供参考，识别结果可能存在误差
    - 如需专业的植物鉴定，请咨询相关专家
    """)
    st.markdown("---")
    st.subheader("反馈与建议")
    feedback = st.text_area("请告诉我们你的使用体验:", height=100)
    if st.button("提交反馈"):
        if feedback:
            st.success("感谢你的反馈！")
        else:
            st.warning("请输入反馈内容")

# 添加自定义CSS
st.markdown("""
<style>
    /* 进度条样式 */
    .progress {
        margin-bottom: 10px;
    }

    /* 整体背景 */
    body {
        background-color: #f9f9f9;
    }

    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }

    /* 标题样式 */
    h1 {
        color: #2c3e50;
        text-align: center;
    }

    /* 副标题样式 */
    h2 {
        color: #34495e;
    }

    /* 按钮样式 */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 5px 15px;
        font-weight: bold;
    }

    /* 文件上传器样式 */
    .stFileUploader>div>div {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)