import numpy as np
from cv2 import cv2
import imutils
from imutils.contours import sort_contours
import tensorflow as tf

print("[INFO] loading handwriting OCR model...")
model = tf.keras.models.load_model('C:/repository/OCR/modelos/digits.model/')

img = cv2.imread('C:/repository/img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)
edged = cv2.Canny(blur,50, 150)
cv2.imshow("edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

chars = [] 

for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)

	if (w >= 8  and h>=15): 
		roi = gray[y:y+h, x:x+w]
		print("thresh img -> w:" + str(roi.shape[1]) + "   h:" + str(roi.shape[0]))
		cv2.imshow("contorno",roi)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		roi = cv2.threshold(roi, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		height, width = roi.shape
		dX = int(max(0, 32 - width) / 2.0)
		dY = int(max(0, 32 - height) / 2.0)
		padded = cv2.copyMakeBorder(roi, top=dY, bottom=dY,
		left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
			value=(255, 255, 255))
			# value=(0, 0, 0))
		padded = cv2.resize(padded, (28, 28))
		print("thresh img -> w:" + str(padded.shape[1]) + "   h:" + str(padded.shape[0]))
		padded = cv2.bitwise_not(padded)
		# cv2.imshow("padded",padded)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)
		padded = padded.reshape(28,28,1)
			
		chars.append( (padded, (x, y, w, h)) )


boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")

preds = model.predict(chars)

labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

for (pred, (x, y, w, h)) in zip(preds, boxes):
	i = np.argmax(pred)
	prob = pred[i]
	label = labelNames[i]

	print("[INFO] {} - {:.2f}%".format(label, prob * 100))


