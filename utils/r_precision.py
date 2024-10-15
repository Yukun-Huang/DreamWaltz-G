from typing import List
import torch
import numpy as np
from PIL import Image


class R_Precision:
    def __init__(self, model_name, device=None, use_diffusers=True) -> None:
        assert model_name in ['B/32', 'B/16', 'L/14']
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.use_diffusers = use_diffusers
        if use_diffusers:
            from transformers import CLIPProcessor, CLIPModel
            model_cards = {
                'B/16': "openai/clip-vit-base-patch16",
                'B/32': "openai/clip-vit-base-patch32",
                'L/14': "openai/clip-vit-large-patch14",
            }
            self.model = CLIPModel.from_pretrained(model_cards[model_name])
            self.processor = CLIPProcessor.from_pretrained(model_cards[model_name])
        else:
            import clip
            model_cards = {
                'B/16': "ViT-B/16",
                'B/32': "ViT-B/32",
                'L/14': "ViT-L/14",
            }
            self.tokenize = clip.tokenize
            self.model, self.preprocess = clip.load(model_cards[model_name], device=device)

    @torch.no_grad()
    def retrieve(self, image: Image.Image, y: int, text_set: List[str]):

        if self.use_diffusers:
            inputs = self.processor(text=text_set, images=image, return_tensors="pt", padding=True)

            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        else:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            text = self.tokenize(text_set).to(self.device)

            # image_features = self.model.encode_image(image)
            # text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy() # [[0.9927937  0.00421068 0.00299572]]
        y_pred = np.argmax(probs)
        return y == y_pred

    @torch.no_grad()
    def retrieve_multi_query(self, images: List[Image.Image], y: int, text_set: List[str]):

        if self.use_diffusers:
            inputs = self.processor(text=text_set, images=images, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        else:
            images = self.preprocess(images).unsqueeze(0).to(self.device)
            text = self.tokenize(text_set).to(self.device)
            logits_per_image, logits_per_text = self.model(images, text)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy() # [[0.9927937  0.00421068 0.00299572]]
        y_pred = np.argmax(probs, axis=-1)
        y_pred = np.argmax(np.bincount(y_pred))
        return y == y_pred
