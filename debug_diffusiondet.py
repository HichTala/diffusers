from functools import partial
from typing import List, Mapping, Union, Any, Tuple

import numpy as np
import torch
import transformers.trainer_utils
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BatchFeature, AutoImageProcessor
import albumentations as A

from src.diffusers.models.diffusiondet.configuration_diffusiondet import DiffusionDetConfig
from src.diffusers.models.diffusiondet.image_processing_diffusiondet import DiffusionDetImageProcessor
from src.diffusers.models.diffusiondet.modeling_diffusiondet import DiffusionDet

def format_image_annotations_as_coco(
        image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
        examples: Mapping[str, Any],
        transform: A.Compose,
        image_processor: AutoImageProcessor,
        return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def collate_fn(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


def main():
    dataset = load_dataset('HichTala/dota')

    image_processor = DiffusionDetImageProcessor()
    train_augment_and_transform = A.Compose(
        [
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=600, p=1.0),
                    A.RandomSizedBBoxSafeCrop(height=600, width=600, p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    train_transform_batch = partial(
        augment_and_transform_batch, image_processor=image_processor, transform=train_augment_and_transform
    )
    dataset['train'] = dataset['train'].with_transform(train_transform_batch)

    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=8,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        sampler=torch.utils.data.RandomSampler(dataset['train']),
        drop_last=False,
        worker_init_fn=transformers.trainer_utils.seed_worker,
        prefetch_factor=2
    )

    config = DiffusionDetConfig()
    model = DiffusionDet(config).cuda()

    for i, batch_sample in enumerate(train_dataloader):
        print(batch_sample["pixel_values"].shape)
        model(batch_sample)


if __name__ == '__main__':
    main()
