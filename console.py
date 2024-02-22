
import cv2
import numpy as np

# Global variables
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial positions
img = None  # Original image
mask = None  # Mask image

def draw(event, x, y, flags, param):
    global ix, iy, drawing, img, mask
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(img, (ix, iy), (x, y), (255, 255, 255), 5)
        cv2.line(mask, (ix, iy), (x, y), (255, 255, 255), 5)
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (255, 255, 255), 5)
        cv2.line(mask, (ix, iy), (x, y), (255, 255, 255), 5)

def mask_save(image_path):
    global img, mask
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found. Please check the path.")
        return
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: 
            cv2.destroyAllWindows()
            break 
        elif k == ord('k'): 
            cv2.imwrite('mask.png', mask)
            print("Mask saved as 'mask.png'.")
            cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "E:/Thesis_Stuffs/Dataset/Mask/38.jpg"
    main(image_path)