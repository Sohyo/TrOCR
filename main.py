from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests


def resize_image(img):
    # Set the desired width (keeping the original aspect ratio)

    original_width, original_height = img.size
    new_width = original_width * 10
    aspect_ratio = original_width / original_height
    new_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_img = img.resize((new_width, new_height))

    # Save the resized image
    resized_img.save("test_images/resized_image.jpg")

# load image from the IAM database (actually this model is meant to be used on printed text)
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
new_image = Image.open('SROIE2019/train/img/X00016469612.jpg', mode='r', formats=None).convert("RGB")
resize_image(new_image)
resized_image = Image.open('test_images/resized_image.jpg', mode='r', formats=None).convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
pixel_values = processor(images=resized_image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)