from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# Modelo (mesma arquitetura do treino)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1) 
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(16*14*14, 2)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Carregar modelo treinado
model = SimpleCNN()
model.load_state_dict(torch.load("breast_model.pth", map_location="cpu"))
model.eval()

# Transformações para imagens recebidas
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"Nenhuma imagem enviada"}), 400
    f = request.files["file"]
    img = Image.open(io.BytesIO(f.read())).convert("L")  #preto e branco
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out,1)
    label = int(pred.item())
    return jsonify({"prediction": "Maligno" if label==1 else "Benigno"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
