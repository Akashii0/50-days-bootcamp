### APPLICATION OF THRESHOLDING: Mainly used in recognizing  patterns and distinguishing between similar features. (recognizing texts,writings,e.t.c)
import cv2

image = cv2.imread("red_lily.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

edge = cv2.Canny(image, 100, 200) # Canny Edge Detection
# res,thres = cv2.threshold(gray, thresh=125, maxval=255, type=0)

#IMG_BLUR = cv2.GaussianBlur(gray,ksize=(9,9),sigmaX=0)

#cv2.imwrite(filename="edge.png", img=edge)
# cv2.imwrite(filename="gray_image.png", img=gray)
# cv2.imwrite(filename="gray_image.png", img=gray)
# cv2.imwrite(filename="image.png", img=gray)
# resize = cv2.resize(gray, dsize=(200,100))



# cv2.imshow(winname="image", )

cv2.imshow("Resize Image", gray)
cv2.waitKey(0)


# print(image)
# print(resize.shape)
# print(type(image))
# cv2.imshow(winname="image", image)





