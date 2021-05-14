import imutils 
import numpy as np
from cv2 import cv2
from align_images import alignImages
from collections import namedtuple
from imutils.contours import sort_contours
import tensorflow as tf

print("[INFO] loading handwriting OCR model...")
model = tf.keras.models.load_model('C:/repository/OCR/modelos/digits.model/')

# print("[INFO] loading and processing images...")
# img = cv2.imread('C:/repository/img.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, 3)

image = cv2.imread("C:\\repository\\OCR\\modelos\\009_recortado.jpg")
template = cv2.imread("C:\\repository\\OCR\\modelos\\Ficha Modelo.jpg")

OCRLocation = namedtuple("OCRLocation", ["id", "bbox","filter_keywords"])

OCR_LOCATIONS = [
    # OCRLocation("Sexo", (58,45,135,22),["sexo"]),
    # OCRLocation("Raca", (218,45,128,22),["raca"]),
    # OCRLocation("Mossa", (378,45,82,25),["mossa"]),
    # OCRLocation("PesoInicial", (85,70,180,23),["pesoinicial"]),
    # OCRLocation("Tetas", (296,70,162,24),["tetas"]),
    # OCRLocation("Galpao", (78,96,122,25),["galpao"]),
    # OCRLocation("Sala", (236,97,128,23),["sala"]),
    # OCRLocation("Baia", (404,97,60,24),["baia"]),
    # OCRLocation("Racao01", (30,205,39,23),["racao01"]),
    OCRLocation("Dia01", (79,205,39,25),["dia01"]),
    OCRLocation("Mes01", (125,206,40,24),["mes01"]),
    OCRLocation("Kg01", (175,205,40,24),["kg01"]),
    # OCRLocation("Racao02", (30,232,39,23),["racao02"]),
    # OCRLocation("Dia02", (79,233,40,24),["dia02"]),
    # OCRLocation("Mes02", (125,233,40,24),["mes02"]),
    # OCRLocation("Kg02", (175,233,40,24),["kg02"]),
    # OCRLocation("Racao03", (30,260,39,23),["racao03"]),
    # OCRLocation("Dia03", (79,261,39,25),["dia03"]),
    # OCRLocation("Mes03", (125,261,40,24),["mes03"]),
    # OCRLocation("Kg03", (175,261,40,24),["kg03"]),
    # OCRLocation("Racao04", (30,289,39,23),["racao04"]),
    # OCRLocation("Dia04", (79,289,39,24),["dia04"]),
    # OCRLocation("Mes04", (125,289,40,24),["mes04"]),
    # OCRLocation("Kg04", (175,289,40,24),["kg04"]),
    # OCRLocation("Racao05", (30,317,39,23),["racao05"]),
    # OCRLocation("Dia05", (79,317,39,24),["dia05"]),
    # OCRLocation("Mes05", (126,317,40,24),["mes05"]),
    # OCRLocation("Kg05", (177,317,39,23),["kg05"]),
    # OCRLocation("Racao06", (30,345,39,23),["racao06"]),
    # OCRLocation("Dia06", (79,344,39,24),["dia06"]),
    # OCRLocation("Mes06", (126,342,40,25),["mes06"]),
    # OCRLocation("Kg06", (177,344,39,23),["kg06"]),
    # OCRLocation("Racao07", (30,373,39,23),["racao07"]),
    # OCRLocation("Dia07", (79,373,39,24),["dia07"]),
    # OCRLocation("Mes07", (126,373,40,24),["mes07"]),
    # OCRLocation("Kg07", (177,373,39,24),["kg07"]),
    # OCRLocation("Racao08", (30,400,40,24),["racao08"]),
    # OCRLocation("Dia08", (79,400,39,24),["dia08"]), 
    # OCRLocation("Mes08", (126,400,40,24),["mes08"]),
    # OCRLocation("Kg08", (177,400,39,24),["kg08"]), 
    # OCRLocation("Racao09", (30,428,40,24),["racao09"]),
    # OCRLocation("Dia09", (79,428,39,24),["dia09"]),
    # OCRLocation("Mes09", (126,428,40,24),["mes09"]),
    # OCRLocation("Kg09", (177,428,39,24),["kg09"]),
    # OCRLocation("Racao10", (30,456,40,24),["racao10"]),
    # OCRLocation("Dia10", (79,456,40,24),["dia10"]),
    # OCRLocation("Mes10", (126,456,40,24),["mes10"]),
    # OCRLocation("Kg10", (177,456,39,24),["kg10"]),
    # OCRLocation("Racao11", (30,483,40,23),["racao11"]),
    # OCRLocation("Dia11", (79,483,40,24),["dia11"]),
    # OCRLocation("Mes11", (126,483,40,24),["mes11"]),
    # OCRLocation("Kg11", (177,483,39,24),["kg11"]),
    # OCRLocation("Racao12", (30,512,40,23),["racao12"]),
    # OCRLocation("Dia12", (79,512,40,24),["dia12"]),
    # OCRLocation("Mes12", (126,512,40,24),["mes12"]),
    # OCRLocation("Kg12", (177,512,39,24),["kg12"]),
]

# align the images
print("[INFO] aligning images...")
aligned = alignImages(image, template)

# loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
	# extract the OCR ROI from the aligned image
	(x, y, w, h) = loc.bbox
	aligned = imutils.resize(aligned, height=640, width=512)
	img = aligned[y:y + h, x:x + w]
	img = cv2.medianBlur(img, 3)
	cv2.imshow("roi",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("[INFO] finding contours...")
	edged = cv2.Canny(img, 30, 150)
	cv2.imshow("roi",edged)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sort_contours(cnts, method="left-to-right")[0]

	chars = []

	print("[INFO] working on letters contours...")
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)

		if (w > 10 and h>=15):
			print("blurred img -> w:" + str(w) + "   h:" + str(h))
			roi = img[y:y+h, x:x+w]
			thresh = cv2.threshold(roi, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			if(w > 28):
				height, width  = thresh.shape
				thresh = cv2.resize(thresh, dsize=(28, height), interpolation=cv2.INTER_CUBIC)
			if(h > 28):
				height, width  = thresh.shape
				thresh = cv2.resize(thresh, dsize=(width, 28),interpolation=cv2.INTER_CUBIC)
			print("threshed img -> w:" + str(thresh.shape[1]) + "   h:" + str(thresh.shape[0]))
			cv2.imshow("contorno",thresh)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			height, width = thresh.shape
			dX = int(max(0, 32 - width) / 2.0)
			dY = int(max(0, 32 - height) / 2.0)

			padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
				left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
				value=(255, 255, 255))
				# value=(0, 0, 0))
			# print("thresh img -> w:" + str(padded.shape[1]) + "   h:" + str(padded.shape[0]))
			padded = cv2.resize(padded, (28, 28))
			# print("thresh img -> w:" + str(padded.shape[1]) + "   h:" + str(padded.shape[0]))
			padded = cv2.bitwise_not(padded)
			cv2.imshow("padded",padded)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
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


