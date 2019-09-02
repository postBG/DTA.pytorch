import numpy as np
from PIL import Image
from torchvision.transforms import RandomHorizontalFlip


class RandomTranslateWithReflect:
    """Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation=14):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new(old_image.mode, (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class RandomTranslationAndHorizontalFlip(object):
    def __init__(self, max_translation=14, flip_probability=0.5):
        self.random_translation_transform = RandomTranslateWithReflect(max_translation)
        self.random_horizontal_flip = RandomHorizontalFlip(flip_probability)

    def __call__(self, img):
        translated = self.random_translation_transform(img)
        return self.random_horizontal_flip(translated)


class RandomTranslation(object):
    def __init__(self, max_translation=2, flip_probability=0.5):
        self.random_translation_transform = RandomTranslateWithReflect(max_translation)

    def __call__(self, img):
        translated = self.random_translation_transform(img)
        return translated
