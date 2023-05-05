#----------Importar Librerias-------------
import numpy as np
import cv2
import pytesseract
from gtts import gTTS
from PIL import Image
import time
from os import remove
#asegurase de Upgrade your versions of pip, setuptools and wheel.
#para instalar playsound instalar la version pip install playsound==1.2.2
from playsound import playsound
#inicializar variables
cuadro=100
anchoCam, altoCam=640,480
#captura de video
cap=cv2.VideoCapture(0)
cap.set(3,anchoCam)
cap.set(4,altoCam)
CTexto=''
#funcion para extraer texto
def Text(image):
    #Funcion para reproducir la voz
    def voz(FileText,lenguaje,nomFile):
        with open(FileText,"r") as lec:
            lectura=lec.read()
        print('4')
        lect=gTTS(text=lectura.lower(),lang=lenguaje,slow=False)
        print('5')
        nombre=nomFile
        remove(nombre)
        lect.save(nombre)
        print('6')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    gris=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    texto=pytesseract.image_to_string(gris)
    print(texto)
    dire=open('Info.txt',"w")
    print('1')
    dire.write(texto)
    print('2')
    dire.close()
    voz("info.txt","es","Voz.mp3")
    print('7')
    audio='Voz.mp3'
    playsound(audio)
    print('8')
#-----------------------------While principal;------------
inicio = time.time()
while True:
    ret,frame=cap.read()
    if ret == False:break
    cv2.putText(frame,"ubique aqui el texto a leer",(158,80),cv2.FONT_HERSHEY_SIMPLEX,0.71,(255,255,0),2)
    cv2.rectangle(frame,(cuadro,cuadro),(anchoCam-cuadro,altoCam-cuadro),(0,0,0),2)
    x1,y1=cuadro,cuadro
    ancho,alto=(anchoCam-cuadro)-x1,(altoCam-cuadro)-y1
    x2,y2=x1+ancho,y1+alto
    doc=frame[y1:y2,x1:x2]
    cv2.imwrite("Imatext.jpg",doc)
    cv2.imshow("lector inteligente",frame)
    t=cv2.waitKey(1)
    if t==27:
        break
    if ((time.time()-inicio)>=2):
        try:
            Text(doc)
        except:
            print("An exception occurred")
        inicio = time.time()
cap.release()
cv2.destroyAllWindows()