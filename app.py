import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # ปิด watcher warning ของ Streamlit

import torch
import streamlit as st
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pytorch_lightning as pl

# ------------------------------------------------------------
# 1) ฟังก์ชันโหลดโมเดล MobileNetV3-Large จากไฟล์ .pt
# ------------------------------------------------------------
@st.cache_resource  # ให้ Streamlit cache โมเดลไว้ ไม่ต้องโหลดซ้ำเมื่อ Refresh
def load_mobilenetv3_model():
    model = models.mobilenet_v3_large(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 7)

    checkpoint_path = r"D:\ProjectMaster\MobileNet\mobilenetv3_large_100_checkpoint_fold0.pt"

    from torch.serialization import add_safe_globals
    add_safe_globals([pl.LightningModule])

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, pl.LightningModule):
        try:
            model = checkpoint.model
        except AttributeError:
            model = checkpoint
    elif isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    return model

# ------------------------------------------------------------
# 2) สร้าง Transform สำหรับเตรียมรูปเข้าระบบ
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------
# 3) ฟังก์ชันทำ Inference และ Clear Cache หลัง Predict
# ------------------------------------------------------------
def predict_image(image: Image.Image, model):
    img_tensor = transform(image).unsqueeze(0)
    print(f"Input tensor shape: {img_tensor.shape}")

    with torch.no_grad():
        outputs = model(img_tensor)
        print(f"Raw model output: {outputs}")

        _, predicted_idx = torch.max(outputs, dim=1)

    idx_to_class = {
        0: "Day 1",
        1: "Day 2",
        2: "Day 3",
        3: "Day 4",
        4: "Day 5",
        5: "Day 6",
        6: "Day 7",
    }
    return idx_to_class[int(predicted_idx)]

# ------------------------------------------------------------
# 4) ส่วนติดต่อผู้ใช้ (UI)
# ------------------------------------------------------------
st.title("Fungi Growth Stage Prediction")
st.write("อัปโหลดรูปเพื่อให้โมเดลทำนาย (Day 1 - Day 7)")

uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)

    if st.button("Predict"):
        # โหลดโมเดล (จะใช้ cache ถ้ายังไม่เคลียร์)
        model = load_mobilenetv3_model()

        # ทำนาย
        prediction = predict_image(image, model)
        st.write(f"## ผลการทำนาย: {prediction}")

        # เคลียร์ cache ของฟังก์ชันโหลดโมเดล เพื่อบังคับโหลดใหม่ในรอบถัดไป
        load_mobilenetv3_model.clear()
