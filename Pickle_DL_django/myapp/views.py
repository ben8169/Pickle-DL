from django.shortcuts import render
from django.http import JsonResponse
import torch
from torchvision import transforms
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity
from .my_models import load_model

model = load_model()
model.eval()  

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)


def predict(request):
    embeddings = None
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_bytes = image_file.read()
        input_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            embeddings = model(input_tensor)
        embeddings_list = embeddings.numpy().tolist()
        return render(request, 'myapp/predict.html', {'embeddings': embeddings_list})
    return render(request, 'myapp/predict.html')