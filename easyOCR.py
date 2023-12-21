from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import imutils
from easyocr import Reader
import cv2
import requests
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from pdf2image import convert_from_path


# 이미지가 리스트 형식으로 들어오거나, 하나만 들어오는 것에 맞춰서 다 보여줄 수 있는 코드
def plt_imshow(title='image', img=None, figsize=(8,5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i+1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()

    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def make_scan_image(image, width, ksize=(5,5), min_threshold=75, max_threshold=200):
    image_list_title = []
    image_list = []

    image = imutils.resize(image, width=width)
    ratio = org_image.shape[1]/float(image.shape[1])

    # 이미지를 grayscale로 변환하고 blur 적용
    # 모서리를 찾기위한 이미지 연산
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    blurred = cv2.GaussianBlur(gray, ksize, 0)
    edged = cv2.Canny(blurred, min_threshold, max_threshold)

    image_list_title = ['gray', 'blurred', 'edged']
    image_list = [gray, blurred, edged]

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None

    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri*0.02, True)

        if len(approx) == 4:
            findCnt = approx
            break

    if findCnt is None:
        raise Exception(('Could not find contours!'))

    print("shape 확인하기")
    print(findCnt.shape)
    checkedshape = findCnt.reshape(4, 2)
    print(checkedshape.shape)
    checkedshapesized = checkedshape * ratio
    print(checkedshapesized.shape)

    output = image.copy()
    cv2.drawContours(output, [findCnt], -1, (0,255,0), 2)

    image_list_title.append('Outline')
    image_list.append(output)

    # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
    transform_image = four_point_transform(org_image, findCnt.reshape(4, 2)*ratio)

    plt_imshow(image_list_title, image_list)
    plt_imshow("Transform", transform_image)

    return transform_image


def putText(cv_img, text, x, y, color=(0,0,0), font_size=11):
    # Colab이 아닌 Local에서 수행 시, gulim.ttc사용

    font = ImageFont.truetype('NanumGothic-Regular.ttf', font_size)
    img = Image.fromarray(cv_img)

    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=color)

    cv_img = np.array(img)

    return cv_img


if __name__ == '__main__':

    '''
    (1254837,)
    <class 'numpy.ndarray'>
    (2479, 1860, 3)
    after original image
    shape 확인하기
    (4, 1, 2)
    (4, 2)
    (4, 2)

    '''


# 문서 crop 없이 바로 OCR 적용
        # 실제로 적용해 보기
    input_path = "transcript_KR"

    dpi = 160
    threshold = 200

    images = convert_from_path(f'test/{input_path}/{input_path}.pdf', dpi=dpi)
    org_image = np.array(images[0])

    for img in images:
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # print(img_np.shape)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # print(gray_img.shape)
        _, document_image = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        # print(thresholded_img.shape)

        # document_image = np.stack((thresholded_img,) * 3, axis=-1)

        plt_imshow("img_np", document_image)

        # document_image = make_scan_image(org_image, width=200, ksize=(5, 5), min_threshold=20, max_threshold=100)

        langs = ['ko', 'en']

        print("[INFO] OCR'ing input image...")

        reader = Reader(lang_list=langs, gpu=True)
        results = reader.readtext(document_image)
        print(f'results = {results}')

        simple_results = reader.readtext(document_image, detail=0)

        for (bbox, text, prob) in results:
            if prob > 0.4:
                print("[INFO] {:.4f}: {}".format(prob, text))

            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            # 추출한 영역에 사각형을 그리고 인식한 글자를 표기합니다.
            cv2.rectangle(document_image, tl, br, (0, 255, 0), 2)
            document_image = putText(document_image, text, tl[0], tl[1] - 100, 0, 50)

        plt_imshow("Image", document_image, figsize=(10, 8))
