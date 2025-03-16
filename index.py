from PIL import Image

import io
import base64

import torch

def image_to_data_url(img, format='png'):
    # 创建一个内存中的字节流
    buffered = io.BytesIO()
    
    # 将PIL.Image对象保存到字节流中，指定图像格式
    img.save(buffered, format=format)
    
    # 从字节流中获取二进制内容
    img_bytes = buffered.getvalue()
    
    # 使用base64对二进制内容进行编码
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # 构建data URL
    data_url = f'data:image/{format};base64,{img_base64}'
    
    return data_url

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TextIteratorStreamer
from threading import Thread
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*400
# max_pixels = 800*1280
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_to_data_url(Image.open("demo5.png"))
                },
            {
                "type": "text",
                "text": "Describe the image content"
                }
            ]
        }
    ]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

streamer = TextIteratorStreamer(
	processor.tokenizer, skip_prompt=True, skip_special_tokens=True
)
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

buffer = ""
for new_text in streamer:
    buffer += new_text
    # Remove <|im_end|> or similar tokens from the output
    buffer = buffer.replace("<|im_end|>", "")

print(buffer)
