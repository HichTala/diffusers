import argparse
import os
from pathlib import Path

import huggingface_hub
import requests
import torch
from tqdm import tqdm

from src.diffusers.models.diffusiondet.modeling_diffusiondet import DiffusionDet
from src.diffusers.models.diffusiondet.configuration_diffusiondet import DiffusionDetConfig

DIFFUSIONDET_REPO = "HichTala/DiffusionDet"


def conv_block_to_diffuser(checkpoint, diffusers_prefix, checkpoint_prefix, block_idx):
    diffusers_checkpoint = {}

    diffusers_checkpoint.update({
        f'{diffusers_prefix}.conv{block_idx}.weight': checkpoint[f'{checkpoint_prefix}.conv{block_idx}.weight'],
        f'{diffusers_prefix}.bn{block_idx}.weight': checkpoint[f'{checkpoint_prefix}.conv{block_idx}.norm.weight'],
        f'{diffusers_prefix}.bn{block_idx}.bias': checkpoint[f'{checkpoint_prefix}.conv{block_idx}.norm.bias'],
        f'{diffusers_prefix}.bn{block_idx}.running_mean': checkpoint[
            f'{checkpoint_prefix}.conv{block_idx}.norm.running_mean'],
        f'{diffusers_prefix}.bn{block_idx}.running_var': checkpoint[
            f'{checkpoint_prefix}.conv{block_idx}.norm.running_var'],
    })

    return diffusers_checkpoint


def diffusiondet_to_diffusers_checkpoint(model, checkpoint):
    diffusers_checkpoint = {}


    diffusers_checkpoint.update({
        "alphas_cumprod": checkpoint["alphas_cumprod"],
        "sqrt_one_minus_alphas_cumprod": checkpoint["sqrt_one_minus_alphas_cumprod"],
        "sqrt_alphas_cumprod": checkpoint["sqrt_alphas_cumprod"],
        "sqrt_recip_alphas_cumprod": checkpoint["sqrt_recip_alphas_cumprod"],
        "sqrt_recipm1_alphas_cumprod": checkpoint["sqrt_recipm1_alphas_cumprod"]
    })

    res_layers_correspondance = {
        "layer1": "res2",
        "layer2": "res3",
        "layer3": "res4",
        "layer4": "res5",
    }

    fpn_inner_blocks_correspondance = {
        0: "fpn_lateral2",
        1: "fpn_lateral3",
        2: "fpn_lateral4",
        3: "fpn_lateral5",

    }
    fpn_layer_blocks_correspondance = {
        0: "fpn_output2",
        1: "fpn_output3",
        2: "fpn_output4",
        3: "fpn_output5",
    }

    # Backbone
    diffusers_checkpoint.update(
        conv_block_to_diffuser(checkpoint, 'backbone._backbone', 'backbone.bottom_up.stem', 1)
    )
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        for idx in range(len(model.backbone._backbone[layer])):
            diffusers_checkpoint.update(
                conv_block_to_diffuser(checkpoint,
                                       f'backbone._backbone.{layer}.{idx}',
                                       f'backbone.bottom_up.{res_layers_correspondance[layer]}.{idx}',
                                       1)
            )
            diffusers_checkpoint.update(
                conv_block_to_diffuser(checkpoint,
                                       f'backbone._backbone.{layer}.{idx}',
                                       f'backbone.bottom_up.{res_layers_correspondance[layer]}.{idx}',
                                       2)
            )
            diffusers_checkpoint.update(
                conv_block_to_diffuser(checkpoint,
                                       f'backbone._backbone.{layer}.{idx}',
                                       f'backbone.bottom_up.{res_layers_correspondance[layer]}.{idx}',
                                       3)
            )
            if idx == 0:
                diffusers_checkpoint.update({
                    f'backbone._backbone.{layer}.{idx}.downsample.0.weight': checkpoint[
                        f'backbone.bottom_up.{res_layers_correspondance[layer]}.{idx}.shortcut.weight'],
                    f'backbone._backbone.{layer}.{idx}.downsample.1.weight': checkpoint[
                        f'backbone.bottom_up.{res_layers_correspondance[layer]}.{idx}.shortcut.norm.weight'],
                    f'backbone._backbone.{layer}.{idx}.downsample.1.bias': checkpoint[
                        f'backbone.bottom_up.{res_layers_correspondance[layer]}.{idx}.shortcut.norm.bias'],
                    f'backbone._backbone.{layer}.{idx}.downsample.1.running_mean': checkpoint[
                        f'backbone.bottom_up.{res_layers_correspondance[layer]}.{idx}.shortcut.norm.running_mean'],
                    f'backbone._backbone.{layer}.{idx}.downsample.1.running_var': checkpoint[
                        f'backbone.bottom_up.{res_layers_correspondance[layer]}.{idx}.shortcut.norm.running_var'],
                })

    for i in range(4):
        diffusers_checkpoint.update({
            f'fpn.inner_blocks.{i}.0.weight': checkpoint[f'backbone.{fpn_inner_blocks_correspondance[i]}.weight'],
            f'fpn.inner_blocks.{i}.0.bias': checkpoint[f'backbone.{fpn_inner_blocks_correspondance[i]}.bias'],
        })
        diffusers_checkpoint.update({
            f'fpn.layer_blocks.{i}.0.weight': checkpoint[f'backbone.{fpn_layer_blocks_correspondance[i]}.weight'],
            f'fpn.layer_blocks.{i}.0.bias': checkpoint[f'backbone.{fpn_layer_blocks_correspondance[i]}.bias'],
        })

    # Head
    head_checkpoint = {k: v for k, v in checkpoint.items() if k.startswith('head')}
    diffusers_checkpoint.update(head_checkpoint)

    return diffusers_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DiffusionDetConfig()
    config.num_labels = 80
    model = DiffusionDet(config)

    orig_weights_path = "https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res50.pth"

    save_path = Path("model.pth")

    try:
        # Download the model file with a progress bar
        with requests.get(orig_weights_path, stream=True) as response:
            response.raise_for_status()  # Ensure the request was successful
            total_size = int(response.headers.get('content-length', 0))  # Get the total file size
            chunk_size = 1024  # Size of each chunk in bytes

            with open(save_path, "wb") as f, tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading orginal model weights",
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        # Load the state dictionary
        state_dict = torch.load(save_path, map_location=device)

        print("converting to diffusers")
        diffusers_checkpoint = diffusiondet_to_diffusers_checkpoint(model, state_dict['model'])
        model.load_state_dict(diffusers_checkpoint)

        # Save the model
        print("Pushing pretrained model to Hugging Face Hub")
        model.push_to_hub(DIFFUSIONDET_REPO)

    finally:
        # Delete the downloaded file
        if save_path.exists():
            os.remove(save_path)
            print("Downloaded weights file deleted.")


if __name__ == "__main__":
    main()
