import pickle
import cv2
import matplotlib.pyplot as plt


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

image = cv2.imread("test1.jpeg")

cv2.imshow("image", image)
cv2.waitKey()
cv2.destroyAllWindows()
