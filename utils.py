# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel
# import os

# dog_list = {
#     "푸들": "static/dog_images/푸들.jpg",
#     "시바견": "static/dog_images/시바견.jpg",
#     "리트리버": "static/dog_images/리트리버.jpg",
#     "시베리안허스키": "static/dog_images/시베리안허스키.jpg",
#     "비숑": "static/dog_images/비숑.jpg",
# }

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # def get_most_similar_dog(user_image_file):
# #     user_image = Image.open(user_image_file).convert("RGB")
# #     dog_images = [Image.open(path).convert("RGB") for path in dog_list.values()]
# #     all_images = [user_image] + dog_images

# #     inputs = processor(images=all_images, return_tensors="pt", padding=True).to(device)
# #     image_features = model.get_image_features(**inputs)

# #     # [0] = user, [1:] = dogs
# #     user_vec = image_features[0].unsqueeze(0)         # shape: (1, D)
# #     dog_vecs = image_features[1:]                     # shape: (N, D)

# #     similarities = torch.nn.functional.cosine_similarity(user_vec, dog_vecs)
# #     max_idx = torch.argmax(similarities).item()
# #     print("similarities:", similarities.tolist())
# #     return list(dog_list.keys())[max_idx]
# # utils.py
# from PIL import Image
# import torch
# from transformers import CLIPProcessor, CLIPModel
# import os

# # 전역 변수 선언만, 로딩은 나중에
# _model = None
# _processor = None
# _device = "cuda" if torch.cuda.is_available() else "cpu"

# def _load_model_and_processor():
#     global _model, _processor
#     if _model is None or _processor is None:
#         _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)

# def get_most_similar_dog(user_image_file):
#     # 최초 호출 시점에만 로드
#     _load_model_and_processor()

#     user_image = Image.open(user_image_file).convert("RGB")
#     dog_images = [Image.open(path).convert("RGB") for path in dog_list.values()]
#     all_images = [user_image] + dog_images

#     inputs = _processor(images=all_images, return_tensors="pt", padding=True).to(_device)
#     image_features = _model.get_image_features(**inputs)
#     user_vec = image_features[0].unsqueeze(0)
#     dog_vecs = image_features[1:]
#     similarities = torch.nn.functional.cosine_similarity(user_vec, dog_vecs)
#     max_idx = torch.argmax(similarities).item()
#     return list(dog_list.keys())[max_idx]
import onnxruntime
import numpy as np
from PIL import Image

dog_list = {
    "푸들": "static/dog_images/푸들.jpg",
    "시바견": "static/dog_images/시바견.jpg",
    "리트리버": "static/dog_images/리트리버.jpg",
    "시베리안허스키": "static/dog_images/시베리안허스키.jpg",
    "비숑": "static/dog_images/비숑.jpg",
}

# onnx 모델 불러오기 (최초 1회)
session = onnxruntime.InferenceSession("clip-vit-base-patch32-image.onnx", providers=["CPUExecutionProvider"])

def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    # CLIP 정규화
    arr = (arr - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr

def get_image_feature(img):
    arr = preprocess(img)
    arr = np.expand_dims(arr, 0)  # batch
    outputs = session.run(["image_embeds"], {"pixel_values": arr})
    return outputs[0][0]  # (512,) vector

def get_most_similar_dog(user_image_file):
    user_img = Image.open(user_image_file).convert("RGB")
    user_feat = get_image_feature(user_img)
    dog_feats = []
    for path in dog_list.values():
        dog_img = Image.open(path).convert("RGB")
        dog_feats.append(get_image_feature(dog_img))
    dog_feats = np.stack(dog_feats)  # (N,512)
    sims = np.dot(dog_feats, user_feat) / (np.linalg.norm(dog_feats, axis=1) * np.linalg.norm(user_feat))
    max_idx = np.argmax(sims)
    return list(dog_list.keys())[max_idx]
