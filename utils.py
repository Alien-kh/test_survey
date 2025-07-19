from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

dog_list = {
    "푸들": "static/dog_images/푸들.jpg",
    "시바견": "static/dog_images/시바견.jpg",
    "리트리버": "static/dog_images/리트리버.jpg",
    "시베리안허스키": "static/dog_images/시베리안허스키.jpg",
    "비숑": "static/dog_images/비숑.jpg",
}

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def get_most_similar_dog(user_image_file):
#     user_image = Image.open(user_image_file).convert("RGB")
#     dog_images = [Image.open(path).convert("RGB") for path in dog_list.values()]
#     all_images = [user_image] + dog_images

#     inputs = processor(images=all_images, return_tensors="pt", padding=True).to(device)
#     image_features = model.get_image_features(**inputs)

#     # [0] = user, [1:] = dogs
#     user_vec = image_features[0].unsqueeze(0)         # shape: (1, D)
#     dog_vecs = image_features[1:]                     # shape: (N, D)

#     similarities = torch.nn.functional.cosine_similarity(user_vec, dog_vecs)
#     max_idx = torch.argmax(similarities).item()
#     print("similarities:", similarities.tolist())
#     return list(dog_list.keys())[max_idx]
# utils.py
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

# 전역 변수 선언만, 로딩은 나중에
_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model_and_processor():
    global _model, _processor
    if _model is None or _processor is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)

def get_most_similar_dog(user_image_file):
    # 최초 호출 시점에만 로드
    _load_model_and_processor()

    user_image = Image.open(user_image_file).convert("RGB")
    dog_images = [Image.open(path).convert("RGB") for path in dog_list.values()]
    all_images = [user_image] + dog_images

    inputs = _processor(images=all_images, return_tensors="pt", padding=True).to(_device)
    image_features = _model.get_image_features(**inputs)
    user_vec = image_features[0].unsqueeze(0)
    dog_vecs = image_features[1:]
    similarities = torch.nn.functional.cosine_similarity(user_vec, dog_vecs)
    max_idx = torch.argmax(similarities).item()
    return list(dog_list.keys())[max_idx]
