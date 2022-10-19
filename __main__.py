import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math


class _coords:
    def __init__(self, coords):
        self.top_left = (int(coords[0][0]), int(coords[0][1]))
        self.top_right = (int(coords[1][0]), int(coords[1][1]))
        self.bottom_right = (int(coords[2][0]), int(coords[2][1]))
        self.bottom_left = (int(coords[3][0]), int(coords[3][1]))
        self.center = (
            int((coords[0][0] + coords[1][0] + coords[2][0] + coords[3][0]) / 4),
            int((coords[0][1] + coords[1][1] + coords[2][1] + coords[3][1]) / 4),
        )
        self.group = 0

    def report(self):
        print(f"{self.top_left}\t{self.top_right}")
        print(f"{self.bottom_left}\t{self.bottom_right}")
        print(f"{self.center}")


def draw_boxes(image):
    image = 0
    image = cv2.imread("test1.jpeg")
    for poly in polys:
        image = cv2.line(image, poly.top_left, poly.top_right, (255, 0, 0), 2)
        image = cv2.line(image, poly.top_right, poly.bottom_right, (255, 0, 0), 2)
        image = cv2.line(image, poly.bottom_right, poly.bottom_left, (255, 0, 0), 2)
        image = cv2.line(image, poly.bottom_left, poly.top_left, (255, 0, 0), 2)
    return image


# finding geometries
# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache,
)

# set image path and export folder directory
image_path = "test1.jpeg"
output_dir = "outputs/"

# read image into cv2
image = read_image(image_path)

# load models
refine_net = load_refinenet_model(
    cuda=False, weight_path="models/craft_refiner_CTW1500.pth"
)
craft_net = load_craftnet_model(cuda=False, weight_path="models/craft_mlt_25k.pth")

# perform prediction
prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    cuda=False,
    long_size=1280,
)

polys = [_coords(coords) for coords in prediction_result["boxes"]]

image = draw_boxes(image)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

empty_cuda_cache()
