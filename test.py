import pickle
import cv2
import matplotlib.pyplot as plt
import math
import __main__


class _coords:
    def __init__(self, coords):
        self.top_left = (int(coords[0][0]), int(coords[0][1]))
        self.top_right = (int(coords[1][0]), int(coords[1][1]))
        self.bottom_right = (int(coords[2][0]), int(coords[2][1]))
        self.bottom_left = (int(coords[3][0]), int(coords[3][1]))
        self.center = (
            abs(self.top_left[0] - self.bottom_right[0]) / 2,
            abs(self.top_left[1] - self.bottom_right[1]) / 2,
        )
        self.group = 0
        self.coords = [
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_right,
        ]

    def report(self):
        print(f"{self.top_left}\t{self.top_right}")
        print(f"{self.bottom_left}\t{self.bottom_right}")
        print(f"{self.center}")

polys = pickle.load(open("polys.pickle", "rb"))

def find_closest(poly):
    points = [[], []]
    closest = 0
    is_first = True
    for point in [poly.top_right, poly.top_left, poly.bottom_right, poly.bottom_left]:
        for comp_poly in polys:
            if comp_poly != poly:
                coords = [comp_poly.top_right, comp_poly.top_left, comp_poly.bottom_right, comp_poly.bottom_left]
                dists = [
                    math.dist(point, comp_poly.top_right),
                    math.dist(point, comp_poly.top_left),
                    math.dist(point, comp_poly.bottom_right),
                    math.dist(point, comp_poly.bottom_left),
                ]
                min_dist = min(dists)
                if min_dist < closest or is_first:
                    is_first = False
                    closest = min_dist
                    points[0] = point
                    points[1] = coords[dists.index(min_dist)]
            else:
                print(f"match found")
    return points

def draw_boxes(image):
    image = 0
    image = cv2.imread("test1.jpeg")
    for poly in polys:
        image = cv2.line(image, poly.top_left, poly.top_right, (255, 0, 0), 2)
        image = cv2.line(image, poly.top_right, poly.bottom_right, (255, 0, 0), 2)
        image = cv2.line(image, poly.bottom_right, poly.bottom_left, (255, 0, 0), 2)
        image = cv2.line(image, poly.bottom_left, poly.top_left, (255, 0, 0), 2)
    return image

image = cv2.imread("test1.jpeg")
image = draw_boxes(image)

for poly in polys:
    closest = find_closest(poly)
    cv2.line(image, closest[0], closest[1], (0, 255, 0), 5)

cv2.imshow("image", image)
cv2.waitKey()
cv2.destroyAllWindows()
