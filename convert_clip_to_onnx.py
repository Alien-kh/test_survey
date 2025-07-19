import torch
from transformers import CLIPProcessor, CLIPModel
import onnx

# 모델과 프로세서 준비
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# 더미 이미지 입력 (1장, 3채널, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# 실제 CLIP은 processor(images=...) → tensor 변환 필요
# 기본적으로 'pixel_values'라는 이름의 입력을 받음
input_names = ["pixel_values"]
output_names = ["image_embeds"]

torch.onnx.export(
    model.vision_model,      # CLIP의 이미지 인코더만 내보냄
    dummy_input,             # 입력 샘플
    "clip-vit-base-patch32-image.onnx",  # 저장 경로
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={"pixel_values": {0: "batch_size"}},
    opset_version=17,
)
print("ONNX 변환 완료!")
