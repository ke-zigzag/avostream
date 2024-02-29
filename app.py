import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
import torch.nn.functional as F
from model import avomodel
from tqdm import tqdm #背景切り取りライブラリー
from rembg import remove
from io import BytesIO

# 画像変換の定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 画像サイズの変更
    transforms.ToTensor(),  # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 標準化
])

#densenetの特徴量をインポート
from torchvision.models import mobilenet_v3_small
feature =  mobilenet_v3_small

#モデルの定義
class Net(pl.LightningModule):

    def __init__(self):
       super().__init__()
       self.feature = mobilenet_v3_small()
       self.fc = nn.Linear(1000, 2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

#モデルインスタンスを作成し、重みを読み込む
avomodel = Net().cpu().eval()
avomodel.load_state_dict(torch.load('mobilenet.pt', map_location=torch.device('cpu')))

# Streamlit UIの構築
st.title("Avocado Checker")

# 画像のアップロード
upload_file = st.file_uploader("Please check your avocado here!", type=["jpg", "png"])

# 画像の推論処理
if upload_file is not None:
    #画像を読み込む
    image_data = upload_file.read()
    image = Image.open(BytesIO(image_data))
    st.image(image, caption="Avocado Image", width=250) #アップロードした画像を表示
    

    #画像をモデルに適した形に変換
    output = remove(image).convert('RGB')
    image_tensor = transform(output)
    
    #予測を実行
    with torch.no_grad():
        prediction = avomodel(image_tensor.unsqueeze(0))
    probabilities = F.softmax(prediction, dim=1)
    predicted_class = torch.argmax(probabilities, 1).item()

    # 推論結果を表示
    if predicted_class == 0:
        st.write("Wait a few days")
    elif predicted_class == 1:
        st.write("Eat me")
    else:
        st.write("Really Avocado??")