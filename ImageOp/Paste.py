
def paste_image_onto_image_at_bbox(base_image, paste_image, bbox):
    base_copy = base_image.copy()
    base_copy[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]] = paste_image
    return base_copy

'''pastes a value into the image at all pixels contained by the bbox'''
def set_bbox_in_image_to_value(image, bbox, value):
    copy_image = image.copy()
    copy_image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]] = value
    return copy_image
