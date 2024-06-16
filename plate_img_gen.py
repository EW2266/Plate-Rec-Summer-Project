from PIL import Image, ImageDraw, ImageFont
import os
from fontTools.ttLib import TTFont

# Path to the Penitentiary Gothic font
font_path = 'font/e-phemera - penitentiary gothic fill.ttf'
font_size = 64  # Adjust font size as needed
save_directory = 'tesseract_training/penitentiary_gothic'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Extract characters from the font file
font = TTFont(font_path)
characters = ''.join(chr(c) for c in font.getBestCmap().keys())

# Generate synthetic images for each character
for i, char in enumerate(characters):
    image = Image.new('RGB', (128, 128), color=(255, 255, 255))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image)
    # Use textbbox to get the bounding box
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((128 - text_width) / 2, (128 - text_height) / 2), char, font=font, fill=(0, 0, 0))
    image.save(os.path.join(save_directory, f'penitentiary_gothic.exp0.{i}.tif'))

    # Create a corresponding box file with UTF-8 encoding
    box_file_path = os.path.join(save_directory, f'penitentiary_gothic.exp0.{i}.box')
    with open(box_file_path, 'w', encoding='utf-8') as box_file:
        box_file.write(f"{char} 0 0 {text_width} {text_height} 0\n")
