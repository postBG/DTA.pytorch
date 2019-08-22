class ToRGB(object):
    def __call__(self, pic):
        return pic.convert('RGB')


class Identity(object):
    def __call__(self, x):
        return x
