"""Small OpenAI-chat-shaped adapter around a Transformers vision-language model."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any


class LocalVisionLanguageClient:
    """Load a Qwen2.5-VL model and optional PEFT adapter for local inference."""

    def __init__(self, model: Any, processor: Any, torch_module: Any):
        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @classmethod
    def from_pretrained(cls, model_name: str, adapter_path: str | None = None):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Local VLM inference requires torch and transformers. Install requirements-training.txt."
            ) from exc

        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Loading a trained adapter requires peft.") from exc
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return cls(model, processor, torch)

    @staticmethod
    def _decode_image(url: str):
        from PIL import Image

        if not url.startswith("data:") or "," not in url:
            raise ValueError("Local VLM backend expects a base64 data URL for the image.")
        encoded = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    def _create(self, model: str, messages: list[dict], temperature: float = 0.0, **kwargs):
        del model
        system = ""
        user_text = ""
        image = None
        for message in messages:
            if message.get("role") == "system":
                system += str(message.get("content", ""))
            elif message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    user_text += content
                else:
                    for part in content:
                        if part.get("type") == "text":
                            user_text += str(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            image = self._decode_image(part["image_url"]["url"])
        if image is None:
            raise ValueError("A meme image is required for local VLM inference.")

        conversation = []
        if system:
            conversation.append({"role": "system", "content": system})
        conversation.append({
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}],
        })
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generation_args = {
            "max_new_tokens": int(kwargs.get("max_tokens", 1024)),
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_args)
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
