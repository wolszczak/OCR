import numpy as np
import imutils
from cv2 import cv2

def alignImages(image, template, maxFeatures=50000, keepPercent=0.3, debug=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keepPercent)
    # print(len(matches))
    # print(keep)
    matches = matches[:keep]

    # if debug:
    #     matchedVis = cv2.drawMatches(
    #         image, kpsA, template, kpsB, matches, None)
    #     matchedVis = imutils.resize(matchedVis,height=640, width=1000)
    #     cv2.imshow("Matched Keypoints", matchedVis)
    #     cv2.waitKey(0)

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(imageGray, H, (w, h))

    if debug:
        aligned = imutils.resize(aligned, height=640, width=512)
        template = imutils.resize(template, height=640,width=512)

        stacked = np.hstack([aligned, template])

        overlay = template.copy()
        output = aligned.copy()
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

        # cv2.imshow("Image Alignment Stacked", stacked)
        cv2.imshow("Image Alignment Overlay", output)
        cv2.waitKey(0)

    return aligned