from Augmentor import Operations, Pipeline
from Augmentor.Operations import *


def crop_center(img):
    width, height = img.size
    if width == height:
        return img

    length = min(width, height)

    left = (width - length) // 2
    upper = (height - length) // 2
    right = left + length
    lower = upper + length

    box = (left, upper, right, lower)
    return img.crop(box)


class CropCenter(Operation):
    """
    Class that allows for a custom operation to be performed using Augmentor's
    standard :class:`~Augmentor.Pipeline.Pipeline` object.
    """
    def __init__(self, probability):
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        """
        Perform the custom operation on the passed image(s), returning the
        transformed image(s).
        :param images: The image to perform the custom operation on.
        :return: The transformed image(s) (other functions in the pipeline
         will expect an image of type PIL.Image)
        """
        augmented_images = []

        for image in images:
            augmented_images.append(crop_center(image))

        return augmented_images


class CustomPipeline(Pipeline):
    def perform_operations(self, image):
        augmented_image = image
        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                augmented_image = operation.perform_operation([augmented_image])[0]
        return augmented_image


def get_augmentation_group(data_aug_group, input_size, center=True, resize=True): 
    resize_prob = 1 if resize else 0
    center_prob = 1 if center else 0

    DATA_AUGMENTATION_GROUPS = [
        # GROUP 0 (NO DATA AUGMENTATION)
        [
            # Center crop
            CropCenter(probability=center_prob),
            # Resize the image to the desired input size 
            Operations.Resize(probability=resize_prob, width=input_size[0], height=input_size[1], resample_filter="BICUBIC")
        ],
        # GROUP 1 (Common transformations: rotations, flips, crops, shears)
        [
            # Rotate the image by 90 degrees randomly
            Operations.Rotate(probability=0.5, rotation=-1),
            # Flip top/bottom
            Operations.Flip(probability=0.5, top_bottom_left_right="TOP_BOTTOM"),
            # Flip left/right
            Operations.Flip(probability=0.5, top_bottom_left_right="LEFT_RIGHT"),
            # Random crops 
            Operations.CropPercentage(probability=0.5, percentage_area=0.85, centre=center, randomise_percentage_area=True),
            # Resize the image to the desired input size 
            Operations.Resize(probability=resize_prob, width=input_size[0], height=input_size[1], resample_filter="BICUBIC")
        ],
        # GROUP 2 (Pixel intensity transformations)
        [
            # Rotate the image by 90 degrees randomly
            Operations.Rotate(probability=0.5, rotation=-1),
            # Flip top/bottom
            Operations.Flip(probability=0.5, top_bottom_left_right="TOP_BOTTOM"),
            # Flip left/right
            Operations.Flip(probability=0.5, top_bottom_left_right="LEFT_RIGHT"),
            # Random change brightness of the image
            Operations.RandomBrightness(probability=0.5, min_factor=0.5,max_factor=1.5),
            # Random change saturation of the image
            Operations.RandomColor(probability=0.5, min_factor=0.5,max_factor=1.5),
            # Random change saturation of the image
            Operations.RandomContrast(probability=0.5, min_factor=0.5, max_factor=1.5),
            # Random crops 
            Operations.CropPercentage(probability=0.5, percentage_area=0.85, centre=center, randomise_percentage_area=True),
            # Resize the image to the desired input size 
            Operations.Resize(probability=resize_prob, width=input_size[0], height=input_size[1], resample_filter="BICUBIC")
        ],
        # GROUP 3 (Perspective transformations)
        [
            # Rotate the image by 90 degrees randomly
            Operations.Rotate(probability=0.5, rotation=-1),
            # Flip top/bottom
            Operations.Flip(probability=0.5, top_bottom_left_right="TOP_BOTTOM"),
            # Flip left/right
            Operations.Flip(probability=0.5, top_bottom_left_right="LEFT_RIGHT"),
            # Shear Image
            Operations.Shear(probability=0.5, max_shear_left=20, max_shear_right=20),
            # Random Tilt up down
            Operations.Skew(probability=0.5, skew_type="TILT_TOP_BOTTOM", magnitude=1.0),
            # Random Tilt left right
            Operations.Skew(probability=0.5, skew_type="TILT_LEFT_RIGHT", magnitude=1.0),
            # Random Skew CORNER
            Operations.Skew(probability=0.5, skew_type="CORNER", magnitude=1.3),
            # Random crops 
            Operations.CropPercentage(probability=0.5, percentage_area=0.85, centre=center, randomise_percentage_area=True),
            # Resize the image to the desired input size 
            Operations.Resize(probability=resize_prob, width=input_size[0], height=input_size[1], resample_filter="BICUBIC")
        ],
        # GROUP 4 (Noise transformations)
        [
            # Center crop
            CropCenter(probability=1),
            # Rotate the image by 90 degrees randomly
            Operations.Rotate(probability=0.5, rotation=-1),
            # Flip top/bottom
            Operations.Flip(probability=0.5, top_bottom_left_right="TOP_BOTTOM"),
            # Flip left/right
            Operations.Flip(probability=0.5, top_bottom_left_right="LEFT_RIGHT"),
            # Random distortions
            Operations.Distort(probability=0.5, grid_width=8, grid_height=8, magnitude=15),
            # Random erasing
            Operations.RandomErasing(probability=0.5, rectangle_area=0.25),
            # Random crops 
            Operations.CropPercentage(probability=0.5, percentage_area=0.85, centre=center, randomise_percentage_area=True),
            # Resize the image to the desired input size 
            Operations.Resize(probability=resize_prob, width=input_size[0], height=input_size[1], resample_filter="BICUBIC")
        ]
    ]

    return DATA_AUGMENTATION_GROUPS[data_aug_group]
