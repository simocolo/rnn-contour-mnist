import cv2
import numpy as np 

ix,iy,drawing = 0,0,False
img = np.zeros((28,28,1), np.uint8)

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(ix,iy),(x,y),255,2)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(ix,iy),(x,y),255,2)
        
    ix,iy=x,y
    return x,y



def draw():
    global img

    cv2.namedWindow('digit', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('digit', 280,280)
    cv2.setMouseCallback('digit',interactive_drawing)

    while(1):
        cv2.imshow('digit',img)
        k=cv2.waitKey(1)&0xFF
        if k!=255:
            break

    cv2.imwrite('img.png', img)
    img = np.zeros((28,28,1), np.uint8)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    draw()