# app.py
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from train import MNISTNet

# Load model
model = MNISTNet()
model.load_state_dict(
    torch.load('model.pth', 
              map_location=torch.device('cpu'),
              weights_only=True)
)
model.eval()

# MNIST classes are digits 0-9
classes = [str(i) for i in range(10)]

def predict(image):
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences = {str(i): float(prob) for i, prob in enumerate(probabilities[0])}
        return confidences

# Create example images with digits
example_images = [
    "examples/digit_0.jpg",
    "examples/digit_1.jpg",
    "examples/digit_2.jpg"
]

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Draw a digit or upload an image"),
    outputs=gr.Label(num_top_classes=3),
    title="MNIST Digit Recognizer",
    description="Draw or upload a hand-written digit (0-9) and the model will predict it!",
    examples=example_images,
    live=True  # Real-time prediction as user draws
)

# Launch the app
if __name__ == "__main__":
    iface.launch()