import cv2
import timm
import torch
from torchvision import transforms

from utils import *

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess function
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = preprocess(frame)
    frame = frame.unsqueeze(0)  # Add batch dimension
    return frame


capture = cv2.VideoCapture(0)


model = timm.create_model(
    config.base_model, pretrained=True, num_classes=config.n_classes
)
model.load_state_dict(torch.load(f"Knife-{config.base_model}-E9.pt"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

cv2.namedWindow("webcam_demo", cv2.WINDOW_NORMAL)

with torch.no_grad():
  while True:
    ret, frame = capture.read()
    if not ret:
      break

    # Preprocess the frame
    input_tensor = preprocess_frame(frame)

    # Move input tensor to the appropriate device
    input_tensor = input_tensor.to(device)

    # Perform inference
    outputs = model(input_tensor)
    preds = outputs.softmax(1)

    # Process the outputs as needed (e.g., display results)

    value, top = preds.topk(1, dim=1, largest=True, sorted=True)
    # Extract the index and value of the top prediction
    top_index = top.item()
    top_value = value.item()

    # Confidence
    if top_value > 0.001:
      cv2.setWindowTitle("webcam_demo", f"Top Prediction: Class {top_index}, Probability: {top_value:.4f}")
    elif cv2.getWindowProperty("webcam_demo", cv2.WND_PROP_VISIBLE) > 0:
      cv2.setWindowTitle("webcam_demo", f"No knives found!")

    # Display the frame
    cv2.imshow('webcam_demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

capture.release()
cv2.destroyAllWindows()