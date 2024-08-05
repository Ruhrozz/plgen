from itertools import chain

import lightning as L
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.nn import functional as F
from transformers import CLIPTextModel
from transformers import CLIPTokenizer


class StableDiffusionModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Initialize models
        name = cfg.model.model_name
        self.unet = UNet2DConditionModel.from_pretrained(name, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(name, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(name, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(name, subfolder="scheduler")

        self.processor = VaeImageProcessor(vae_scale_factor=0.18215)  # pyright: ignore

        self.save_hyperparameters()

        # Do not train anything except unet
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)  # pyright: ignore

    def get_tokens(self, caption, tokenizer):
        text_inputs = tokenizer(
            caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        return text_inputs.input_ids.to(self.device)

    def encode_tokenized_text(self, tokens, text_encoder):
        attention_mask = None
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = tokens.attention_mask.to(self.device)

        return text_encoder(tokens, attention_mask=attention_mask).last_hidden_state

    @staticmethod
    def gen_timesteps(noise_scheduler, batch_size):
        return torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,)).long()

    def training_step(self, batch, batch_idx):  # pylint: disable=W0221 # Variadics removed in overriding
        del batch_idx

        image, caption = batch

        # Get text embeddings
        tokens = self.get_tokens(caption=caption, tokenizer=self.tokenizer)
        text_embeds = self.encode_tokenized_text(tokens=tokens, text_encoder=self.text_encoder)

        # Encode VAE
        latents = self.vae.encode(image).latent_dist.sample()  # pyright: ignore
        latents = latents * 0.18215

        # Add noise
        timesteps = self.gen_timesteps(self.noise_scheduler, batch_size=latents.shape[0]).to(self.device)
        noise = torch.randn_like(latents).to(self.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        pred_noise = self.unet(noisy_latents, timesteps, text_embeds).sample  # pyright: ignore
        loss = F.mse_loss(pred_noise, noise)

        self.log("loss/noise", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):  # pyright: ignore
        parameters = self.unet.parameters()  # pyright: ignore
        if self.cfg.model.train_text_encoder:
            parameters = chain(parameters, self.text_encoder.parameters())

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        scheduler = get_scheduler(
            name=self.cfg.optim.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.trainer_params.max_steps * self.cfg.optim.warmup_relative,
            num_training_steps=self.cfg.optim.scheduler_max_steps or self.cfg.trainer_params.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
