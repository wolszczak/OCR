from align_images import alignImages
from collections import namedtuple
import numpy as np
import imutils
from cv2 import cv2

image = cv2.imread("C:\\repository\\OCR\\modelos\\009_recortado.jpg")
template = cv2.imread("C:\\repository\\OCR\\modelos\\Ficha Modelo.jpg")

OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
	"filter_keywords"])

OCR_LOCATIONS = [
    OCRLocation("Sexo", (58,45,135,22),["sexo"]),
    OCRLocation("Raca", (218,45,128,22),["raca"]),
    OCRLocation("Mossa", (378,45,82,25),["mossa"]),
    OCRLocation("PesoInicial", (85,70,180,24),["pesoinicial"]),
    OCRLocation("Tetas", (296,70,162,24),["tetas"]),
    OCRLocation("Galpao", (78,96,122,25),["galpao"]),
    OCRLocation("Sala", (236,97,128,23),["sala"]),
    OCRLocation("Baia", (404,97,60,24),["baia"]),
    # OCRLocation("Racao01", (30,205,39,23),["racao01"]),
    # OCRLocation("Dia01", (79,205,39,25),["dia01"]),
    # OCRLocation("Mes01", (125,206,40,24),["mes01"]),
    # OCRLocation("Kg01", (175,205,40,24),["kg01"]),
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


print("[INFO] OCR-ing document...")
parsingResults = []

# loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
	# extract the OCR ROI from the aligned image
    (x, y, w, h) = loc.bbox
    aligned = imutils.resize(aligned, height=640, width=512)
    roi = aligned[y:y + h, x:x + w] 
    # template = imutils.resize(template,height=640,width=512)
    # roi = template[y:y + h, x:x + w]
    # cv2.imwrite("C:/repository/img.jpg",roi)
    cv2.imshow("roi",roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

	# # OCR the ROI using TesseRacaot
    # rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # text = pytesseRacaot.image_to_string(rgb)

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

# # RacaoAO COLUNA 1
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

# # RacaoAO COLUNA 2
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
