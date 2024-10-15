from typing import List
from transformers import AutoTokenizer, CLIPTextModel


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_type='L/14') -> None:
        super().__init__()
        model_cards = {
            'B/16': "openai/clip-vit-base-patch16",
            'B/32': "openai/clip-vit-base-patch32",
            'L/14': "openai/clip-vit-large-patch14",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_cards[model_type])
        self.text_model = CLIPTextModel.from_pretrained(model_cards[model_type])

    def forward(self, text_list: List[str]):
        inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        outputs = self.text_model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # [B, T, 768]
        pooled_output = outputs.pooler_output  # [B, 768], pooled (EOS token) states
        return last_hidden_state, pooled_output
