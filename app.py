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
    # 1. โหลดโมเดล MobileNetV3-Large แบบไม่มี pretrained weights
    model = models.mobilenet_v3_large(pretrained=False)
    
    # 2. ปรับ final layer ให้มีจำนวนคลาสตรงกับที่เทรน (7 คลาส)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 7)  # 7 คลาส: Day 1 - Day 7
    
    # 3. โหลด checkpoint (full LightningModule, ไม่ใช่แค่ state dict)
    checkpoint_path = r"MobileNetV3/mobilenetv3_large_100_checkpoint_fold4.pt"
    
    # 4. ใช้ add_safe_globals เพื่อโหลด checkpoint โดยไม่มีปัญหากับ unsafe globals
    from torch.serialization import add_safe_globals
    add_safe_globals([pl.LightningModule])  # Safe loading of LightningModule
    
    # 5. โหลด checkpoint โดยใช้ torch.load
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 6. ตรวจสอบว่า checkpoint เป็น Lightning module และดึง model ออกมา
    if isinstance(checkpoint, pl.LightningModule):
        model = checkpoint.model  # Extract the model from the checkpoint
    
    # 7. ตั้งโมเดลเป็นโหมดประเมินผล
    model.eval()
    
    return model

# โหลดโมเดล MobileNetV3-Large
model = load_mobilenetv3_model()

# ------------------------------------------------------------
# 2) สร้าง Transform สำหรับเตรียมรูปเข้าระบบ
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ปรับขนาดรูป
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------
# 3) ฟังก์ชันทำ Inference (ทำนายคลาสจากรูปภาพ)
# ------------------------------------------------------------
def predict_image(image: Image.Image, model: nn.Module):
    img_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]
    
    with torch.no_grad():
        outputs = model(img_tensor)  # => shape [1, 7]
        _, predicted_idx = torch.max(outputs, dim=1)
    
    # Map index -> class name
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
# 4) ส่วนติดต่อผู้ใช้ (UI) ด้วย Streamlit
# ------------------------------------------------------------
st.title("Fungi Growth Stage Prediction")
st.write("อัปโหลดรูปเพื่อให้โมเดลทำนาย (Day 1 - Day 7)")

uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
    
    if st.button("Predict"):
        prediction = predict_image(image, model)
        st.write(f"## ผลการทำนาย: {prediction}")
