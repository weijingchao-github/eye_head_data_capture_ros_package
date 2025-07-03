import cv2


def main():
    image_path = ""
    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
