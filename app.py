import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from model import MyLightningModel  # import โมเดลของเรา

# โหลดโมเดลแบบ Lightning full checkpoint
@st.cache_resource
def load_mobilenetv3_model():
    checkpoint_path = r"MobileNetV3/mobilenetv3_large_100_checkpoint_fold4.pt"
    model = MyLightningModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

model = load_mobilenetv3_model()

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
        prediction = predict_image(image, model)
        st.write(f"## ผลการทำนาย: {prediction}")
