
def crop_image_to_bbox(image, bbox):
    return image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
