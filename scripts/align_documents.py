from align_images import alignImages
from collections import namedtuple
import pytesseract
import numpy as np
import imutils
from cv2 import cv2

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
	"filter_keywords"])

OCR_LOCATIONS = [
    # OCRLocation("Sexo", (57,43,135,28),["sexo"]),
    # OCRLocation("Raca", (219,43,128,28),["raca"]),
    OCRLocation("Mossa", (378,45,83,23),["mossa"])

]

image = cv2.imread("C:\\repository\\OCR\\modelos\\009_recortado.jpg")
template = cv2.imread("C:\\repository\\OCR\\modelos\\Ficha Modelo.jpg")

# align the images
print("[INFO] aligning images...")
aligned = alignImages(image, template)


print("[INFO] OCR-ing document...")
parsingResults = []

# loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
	# extract the OCR ROI from the aligned image
    (x, y, w, h) = loc.bbox
    aligned = imutils.resize(aligned, height=640, width=512)
    roi = aligned[y:y + h, x:x + w]
    cv2.imwrite("C:/repository/img.jpg",roi)
    cv2.imshow("roi",roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

	# # OCR the ROI using Tesseract
    # rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # text = pytesseract.image_to_string(rgb)

    # print(text)

    # cv2.imshow("roi",roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()








# # CABECALHO

# # LINHA 1
# cv2.rectangle(template, (57,69), (190,43),(0, 255, 0), 2)
# cv2.rectangle(template, (216,69), (344,43),(0, 255, 0), 2)
# cv2.rectangle(template, (375,69), (460,43),(0, 255, 0), 2)
# # LINHA 2
# cv2.rectangle(template, (85,95), (269,66),(0, 255, 0), 2)
# cv2.rectangle(template, (295,95), (460,66),(0, 255, 0), 2)
# # LINHA 3
# cv2.rectangle(template, (78,120), (200,92),(0, 255, 0), 2)
# cv2.rectangle(template, (235,120), (365,92),(0, 255, 0), 2)
# cv2.rectangle(template, (400,120), (460,92),(0, 255, 0), 2)

# # RACAO COLUNA 1
# cv2.rectangle(template, (30,230), (214,200),(0, 255, 0), 2)
# cv2.rectangle(template, (30,257), (214,227),(0, 255, 0), 2)
# cv2.rectangle(template, (30,284), (214,254),(0, 255, 0), 2)
# cv2.rectangle(template, (30,313), (214,282),(0, 255, 0), 2)
# cv2.rectangle(template, (30,340), (214,310),(0, 255, 0), 2)
# cv2.rectangle(template, (30,367), (214,337),(0, 255, 0), 2)
# cv2.rectangle(template, (30,395), (214,365),(0, 255, 0), 2)
# cv2.rectangle(template, (30,422), (214,392),(0, 255, 0), 2)
# cv2.rectangle(template, (30,450), (214,419),(0, 255, 0), 2)
# cv2.rectangle(template, (30,477), (214,447),(0, 255, 0), 2)
# cv2.rectangle(template, (30,504), (214,474),(0, 255, 0), 2)
# cv2.rectangle(template, (30,534), (214,501),(0, 255, 0), 2)

# # RACAO COLUNA 2
# cv2.rectangle(template, (222,230), (412,200),(0, 255, 0), 2)
# cv2.rectangle(template, (222,257), (412,227),(0, 255, 0), 2)
# cv2.rectangle(template, (222,284), (412,254),(0, 255, 0), 2)
# cv2.rectangle(template, (222,313), (412,282),(0, 255, 0), 2)
# cv2.rectangle(template, (222,340), (412,310),(0, 255, 0), 2)
# cv2.rectangle(template, (222,367), (412,337),(0, 255, 0), 2)
# cv2.rectangle(template, (222,395), (412,365),(0, 255, 0), 2)
# cv2.rectangle(template, (222,422), (412,392),(0, 255, 0), 2)
# cv2.rectangle(template, (222,450), (412,419),(0, 255, 0), 2)
# cv2.rectangle(template, (222,477), (412,447),(0, 255, 0), 2)
# cv2.rectangle(template, (222,504), (412,474),(0, 255, 0), 2)
# cv2.rectangle(template, (222,534), (412,501),(0, 255, 0), 2)

# # RODAPE
# cv2.rectangle(template, (153,561), (453,530),(0, 255, 0), 2)
# cv2.rectangle(template, (51,584), (453,558),(0, 255, 0), 2)


# # cv2.imwrite("C:/repository/OCR/modelos/Campos Leitura.jpg",template)
# cv2.imshow("TEMPLATE",template)


# cv2.waitKey(0)
