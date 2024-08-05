import os

import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore

from base_diffusion import DiffusionModel
from utils.utils import pil_hstack


class CLIPScoreDiffusionModel(DiffusionModel):
    def on_predict_start(self):
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(
            self.device,
            dtype=self.dtype,
        )
        self.clip_score = []

        if self.conf.model.cache_interval:
            from DeepCache import DeepCacheSDHelper

            helper = DeepCacheSDHelper(pipe=self.pipe)
            helper.set_params(
                cache_interval=self.conf.model.cache_interval,
                cache_branch_id=0,
            )
            helper.enable()

    def predict_step(self, batch, batch_idx):
        real_images, text = batch

        gen_image = self.pipe(
            list(text),
            negative_prompt=[self.negative_prompt for _ in text],
            num_inference_steps=self.num_inference_steps,
        ).images

        gen_image_tensor = torch.Tensor(
            [np.asarray(img).astype("uint8") for img in gen_image],
        ).permute(0, -1, 1, 2)

        score = self.clip(
            gen_image_tensor.to(self.device, dtype=torch.uint8),
            list(text),
        )
        self.clip_score.append(score.cpu().detach().item())

        if batch_idx < self.conf.logging.fid_images:
            img1 = real_images[0].detach().cpu().numpy()
            img1 = Image.fromarray(img1.transpose(1, 2, 0).astype(np.uint8))
            img2 = gen_image[0]
            cat_img = pil_hstack([img1, img2])
            cat_img.save(self.image_output_dir + f"/{batch_idx}.png")
            with open(self.image_output_dir + f"/prompts.txt", "a") as file:
                file.write(text[0] + "\n")

    def on_predict_end(self):
        with open(self.logger.log_dir + "/eval_result.txt", "w") as file:
            file.write(f"CLIPScore: {np.mean(self.clip_score)}")
