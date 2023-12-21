# easyOCR
 easyOCR test

## 1. Install easyOCR   
```
pip install easyOCR imutils matplotlib pdf2image PIL opencv-python
```

## 2.  easyOCR과 pytesseract랑의 차이
- 영어, 한국어 지정하지 않아도 인식이 가능하다.
- 한글 인식률이 더 높다.
- 이미지 내의 문자를 찾아서 인식한다.

## 3. PINO 문서 테스트  

- 피노 문서는 인증 도장이 찍혀서 나오는데, image transform 할 때, 도장이 찍힌 부분이 인식이 되어 버린다.
- 이후, image crop 할 때, 모양이 이상함.

## 4. image transform 없이 바로 easyOCR 적용  

- 일단, 한국어는 괜찮은데 영어 인식률이 떨어진다. 


easyOCR 4번 pool로 돌렸을 때  
텍스트 추출 끝: 0:01:25.927955
it took 0:01:44.342340 seconds.

pytesseract 4번 pool로 돌렸을 때      
텍스트 추출 끝: 3.1203503608703613
총 소요시간: 36.68484830856323 seconds 

