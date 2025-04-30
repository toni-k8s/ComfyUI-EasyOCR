import folder_paths
import json
import cv2
import easyocr
import os
import logging
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger("ComfyUI-EasyOCR")
model_dir_name = "EasyOCR"

lang_list = {
    "English": "en",
    "简体中文": "ch_sim",
    "繁體中文": "ch_tra",
    "العربية": "ar",
    "Azərbaycan": "az",
    "Euskal": "eu",
    "Bosanski": "bs",
    "Български": "bg",
    "Català": "ca",
    "Hrvatski": "hr",
    "Čeština": "cs",
    "Dansk": "da",
    "Nederlands": "nl",
    "Eesti": "et",
    "Suomi": "fi",
    "Français": "fr",
    "Galego": "gl",
    "Deutsch": "de",
    "Ελληνικά": "el",
    "עברית": "he",
    "हिन्दी": "hi",
    "Magyar": "hu",
    "Íslenska": "is",
    "Indonesia": "id",
    "Italiano": "it",
    "日本語": "ja",
    "한국어": "ko",
    "Latviešu": "lv",
    "Lietuvių": "lt",
    "Македонски": "mk",
    "Norsk": "no",
    "Polski": "pl",
    "Português": "pt",
    "Română": "ro",
    "Русский": "ru",
    "Српски": "sr",
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "Español": "es",
    "Svenska": "sv",
    "ไทย": "th",
    "Türkçe": "tr",
    "Українська": "uk",
    "Tiếng Việt": "vi",
}

def get_lang_list():
    result = []
    for key, value in lang_list.items():
        result.append(key)
    return result


def get_classes(label):
    label = label.lower()
    labels = label.split(",")
    result = []
    for l in labels:
        for key, value in lang_list.items():
            if l == value:
                result.append(value)
                break
    return result


def get_classes2(label):
    label = label.lower()
    labels = label.split(",")
    result = []
    for l in labels:
        for key, value in lang_list.items():
            if l == key:
                result.append(value)
                break
    return result


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    result = tgt["result"]

    box_color = (255, 0, 0)  # Red color for the box
    text_color = (255, 255, 255)  # White color for the text

    draw = ImageDraw.Draw(image_pil)

    font_path = r"/root/autodl-tmp/ComfyUI/custom_nodes/comfyui-easyocr/docs/PingFangRegular.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }

    # 初始化一个全零的单一遮罩
    merged_mask = np.zeros((H, W, 1), dtype=np.uint8)

    for item in result:
        formatted_points, label, threshold = item

        x1, y1 = formatted_points[0]
        x2, y2 = formatted_points[2]
        threshold = round(threshold, 2)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        points = [[x1, y1], [x2, y2]]

        # 保存 labelme 标注
        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "threshold": str(threshold)
        }
        labelme_data["shapes"].append(shape)

        # 绘制图像中的框与文字
        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([(x1, y1 - text_height - 10), (x1 + text_width, y1)], fill=box_color)
        draw.text((x1, y1 - text_height - 10), f"{label}:{threshold}", font=font, fill=text_color)

        # 在 merged_mask 中叠加矩形区域
        cv2.rectangle(merged_mask, (x1, y1), (x2, y2), (255,), thickness=-1)

    # 将 merged_mask 转为 tensor 格式
    mask_tensor = torch.from_numpy(merged_mask).permute(2, 0, 1).float() / 255.0

    # 图像处理
    image_with_boxes = np.array(image_pil)
    image_with_boxes_tensor = torch.from_numpy(image_with_boxes.astype(np.float32) / 255.0)
    image_with_boxes_tensor = torch.unsqueeze(image_with_boxes_tensor, 0)

    return [image_with_boxes_tensor], [mask_tensor], labelme_data


class ApplyEasyOCR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gpu": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "detect": (
                    ["choose", "input"],
                    {"default": "choose"},
                ),
                "language_list": (
                    get_lang_list(),
                    {"default": "English"},
                ),
                "language_name": (
                    "STRING",
                    {"default": "ch_sim,en", "multiline": False},
                ),
            },
        }

    CATEGORY = "ComfyUI-EasyOCR"
    FUNCTION = "main"
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "JSON",
    )

    def main(self, image, gpu, detect, language_list, language_name):
        res_images = []
        res_masks = []
        res_labels = []

        for item in image:
            image_pil = Image.fromarray(np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")

            language = None
            if detect == "choose":
                language = get_classes2(language_list)
            else:
                language = get_classes(language_name)

            model_storage_directory = os.path.join(folder_paths.models_dir, model_dir_name)
            if not os.path.exists(model_storage_directory):
                os.makedirs(model_storage_directory)

            reader  = easyocr.Reader(language, model_storage_directory=model_storage_directory,gpu=gpu)
            result = reader.readtext(np.array(image_pil))

            size = image_pil.size
            pred_dict = {
                "size": [size[1], size[0]],
                "result":result
            }

            image_tensor, mask_tensor, labelme_data = plot_boxes_to_image(image_pil, pred_dict)

            res_images.extend(image_tensor)
            res_masks.extend(mask_tensor)
            res_labels.append(labelme_data)

            if len(res_images) == 0:
                res_images.extend(item)
            if len(res_masks) == 0:
                mask = np.zeros((height, width, 1), dtype=np.uint8)
                empty_mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
                res_masks.extend(empty_mask)

        return (
            torch.cat(res_images, dim=0),
            torch.cat(res_masks, dim=0),
            res_labels,
        )
