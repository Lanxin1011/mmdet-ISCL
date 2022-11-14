import cv2

def draw_text(img, text, position, color=(0, 0, 255)):
    if not isinstance(text, str):
        text = "{:.2f}".format(text)
    cv2.putText(img, text, (int(position[0]), int(position[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=color, thickness=2, lineType=8)

    return img

def draw_point(img, point, size=3, color=(0, 0, 255)):
    cv2.circle(img, center=(int(point[0]), int(point[1])), radius=size, thickness=-1, color=color)
    
    return img