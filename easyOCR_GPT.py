import datetime
from multiprocessing import Pool

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


def easyocr_run(input_path, dpi, threshold):
    images = convert_from_path(f'test/{input_path}/{input_path}.pdf', dpi=dpi)

    full_string = ""

    for img in images:
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, document_image = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

        langs = ['ko', 'en']

        print("[INFO] OCR'ing input image...")

        reader = Reader(lang_list=langs, gpu=True)
        simple_results = reader.readtext(document_image, detail=0)

        full_string += " ".join(simple_results)

    return full_string


def gpt_api(api_key, text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "please check if the below data is graduation certificate or transcript."
                                "The data is the combination of data extracted from images using OCR four times. There "
                                "are some repetitions and some errors as each OCR creates different results. Please "
                                "consider that this document is transcript or graduation certificate and rearrange only"
                                " considerably important data into json format."
                                "If the data is about transcript, the following data are necessary: name,"
                                " student number, name of university, data of admission, department, date of birth, "
                                "major, date of graduation, courses/ grade/ credits per semester, total credits, "
                                "and accumulative grades. The rest of the data should go to others part of json. "
                                "Choose the most appropriate text from the four OCR results. Revision can be made "
                                "considering this is transcript or graduation certificate since this is extracted "
                                "text from OCR. please provide the json in the language the text was written"
                    },
                    {
                        "type": "text",
                        "text": f"the data is [{text}]"
                    }
                ]
            }
        ],
        "max_tokens": 4000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    data = response.json()
    result = data['choices'][0]['message']['content']

    return result


if __name__ == '__main__':

    start = datetime.datetime.now()
    print("start")

    input_path = "transcript_KR"

    threshold = 200
    dpi = 180
    # dpi_values = [180, 190, 200, 230, 250]
    #


    # params_list = [(input_path, dpi, threshold) for dpi in dpi_values]
    #
    # with Pool(5) as p:
    #     extracted_text = p.starmap(easyocr_run, params_list)

    extracted_text = easyocr_run(input_path, dpi, threshold)

    text_for_gpt = " ".join(extracted_text)
    print(text_for_gpt)

    middle = datetime.datetime.now()
    print(f"텍스트 추출 끝: {middle-start}")

    with open("../api_key.txt", 'r', encoding='utf-8') as file:
        api_key = file.readline()

    text_in_json = gpt_api(api_key, text_for_gpt)
    print(text_in_json)

    end = datetime.datetime.now()
    print(f"it took {end-start} seconds.")

    # for (bbox, text, prob) in results:
    #     if prob > 0.4:
    #         print("[INFO] {:.4f}: {}".format(prob, text))

