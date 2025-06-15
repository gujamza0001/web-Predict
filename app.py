import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from model import MyLightningModel
import os

# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_mobilenetv3_model(checkpoint_path):
    try:
        model = MyLightningModel.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None

# UI สำหรับอัปโหลด checkpoint ถ้าไม่เจอไฟล์
DEFAULT_CHECKPOINT_PATH = "MobileNetV3/mobilenetv3_large_100_checkpoint_fold4.pt"
checkpoint_path = DEFAULT_CHECKPOINT_PATH

if not os.path.exists(checkpoint_path):
    st.warning("ไม่พบไฟล์ checkpoint โมเดล อัปโหลดไฟล์ .pt ได้ที่นี่")
    uploaded_ckpt = st.file_uploader("อัปโหลด checkpoint (.pt)", type=["pt"])
    if uploaded_ckpt is not None:
        checkpoint_path = f"temp_uploaded_ckpt.pt"
        with open(checkpoint_path, "wb") as f:
            f.write(uploaded_ckpt.read())
else:
    st.info(f"ใช้ checkpoint: {checkpoint_path}")

model = load_mobilenetv3_model(checkpoint_path)

# เตรียม transform สำหรับรูปภาพ
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image: Image.Image, model):
    img_tensor = transform(image).unsqueeze(0)  # เพิ่ม batch dim
    with torch.no_grad():
        outputs = model(img_tensor)  # shape [1,7]
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

# Streamlit UI
st.title("Fungi Growth Stage Prediction")
st.write("อัปโหลดรูปเพื่อให้โมเดลทำนาย (Day 1 - Day 7)")

uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
    if st.button("Predict"):
        if model is None:
            st.error("ไม่สามารถทำนายได้ เพราะโหลดโมเดลไม่สำเร็จ")
        else:
            prediction = predict_image(image, model)
            st.write(f"## ผลการทำนาย: {prediction}")
