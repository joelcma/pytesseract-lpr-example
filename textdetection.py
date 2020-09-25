import cv2
import pytesseract
import argparse
import sys
import glob
import os
import time
import numpy as np
import imutils

current_image_name = ''

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '--image', required=False, type=str, help="The image to run tesseract OCR on")
    parser.add_argument('-imgs', '--folder', required=False, type=str, help="The folder to look for images to run pytesseract OCR on")
    args = parser.parse_args()
    if args.image is None and args.folder is None:
        parser.print_help()
        sys.exit(0)
    return args

def main():
    global current_image_name
    args = getArgs()
    if args.image:
        current_image_name = args.image.split('/')[-1]
        # Perform OCR on the given image
        readImgAndDoOCR(args.image)
    else:
        # Read all file paths in folder
        images = glob.glob(os.path.join(args.folder, '*.jpeg'))
        # Perform OCR on all the images
        for path in images:
            current_image_name = path.split('/')[-1]
            readImgAndDoOCR(path)

def readImgAndDoOCR(path):
    img = cv2.imread(path)
    ocr_utput, total_time, cleaning_time, ocr_time = doOCR(img)
    (H, W) = img.shape[:2]
    print(f'Resolution: {W}x{H}\nCleaning: {cleaning_time} ms\nOcr: {ocr_time} ms\nTotal: {total_time} ms\n{path} -> "{ocr_utput}"')
    print("")

def doOCR(img):
    t1 = time.time() # Timing
    img, save_time = cleanImg(img) # Cleaning
    t2 = time.time() # Timing
    output = pytesseract.image_to_string(img) # OCR
    t3 = time.time() # Timing
    if bad(output):
        img = cv2.bitwise_not(img)
        t31 = time.time()
        cv2.imwrite(f'./cleaning_stages/{current_image_name}_5.jpeg', img)
        t32 = time.time()
        save_time += t32-t31
        pytesseract.image_to_string(img)
    t4 = time.time() # Timing
    return output, int((t4-t1 - save_time)*1000), int((t2-t1)*1000), int((t3-t2)*1000)

def bad(output):
    if len(output) < 4:
        return True
    return False

def cleanImg(img):
    # Clean image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grey scale
    blur = cv2.bilateralFilter(gray, 11, 17, 17) # Blur to reduce noise
    thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cleaned = character_isolation(thresholded)

    # Write cleaned images to cleaning_stages folder
    t1 = time.time()
    cv2.imwrite(f'./cleaning_stages/{current_image_name}_1.jpeg', gray)
    cv2.imwrite(f'./cleaning_stages/{current_image_name}_2.jpeg', blur)
    cv2.imwrite(f'./cleaning_stages/{current_image_name}_3.jpeg', thresholded)
    cv2.imwrite(f'./cleaning_stages/{current_image_name}_4.jpeg', cleaned)
    t2 = time.time()
    save_time = t2-t1 # Subtract this from the total computation time.
    return cleaned, save_time

def character_isolation(img):
    # find contours in the thresholded image and sort them by
    # their size in descending order, keeping only the largest
    # ones
    keep = 10
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:keep] # Skip the first contour because it's the whole plate

    # Create blank image with same background color as the given image
    (H,W) = img.shape[:2]
    new_img = get_background_color(img) * np.ones(shape=[H,W], dtype=np.uint8)

    count = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 15 or h < 15: # 15 pixels width or height is definitely junk
            continue
        
        contour_crop = img[y:y+h, x:x+w]
        new_img[y:y+h, x:x+w] = contour_crop
        count += 1
    
    return new_img

def get_background_color(img):
    (H, W) = img.shape[:2]

    resolution = 10 # 1/resolution
    color_count = {}
    for x in range(int(W/resolution)):
        for y in range(int(H/resolution)):
            color = img[y*resolution][x*resolution]
            if color in color_count:
                color_count[color] += 1
            else:
                color_count[color] = 1

    bg_color = 0
    bg_count = 0
    for color, count in color_count.items():
        if count > bg_count:
            bg_color = color

    return bg_color

main()