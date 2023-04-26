#----------Importar Librerias-------------
import numpy as np
import cv2
import pytesseract
from PIL import Image
#realizamos captura de video
cap=cv2.VideoCapture(0)

CTexto=''
#
def texto(image):
    #declaramos la direccion de p
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #extraemos el texto
    config="--psm 1"
    texto=pytesseract.image_to_string(image,config=config)
    return texto

while True:
    ret,frame=cap.read()
    #se voltea el frame para poder leerlo en vertical
    frame=cv2.flip(frame,0)
    frame=cv2.transpose(frame)
    #si no hay retorno de imagen se apaga
    if(ret==False):
        break
    #dibujamos un rectangulo
    cv2.rectangle(frame,(50,550),(400,640),(0,0,0),cv2.FILLED)
    cv2.putText(frame,CTexto[0:7],(50,550),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    #EXTRAEMOS EL ANCHO Y ALTO DEL FOTOGRAMA
    wid,he,c=frame.shape
    #TOMAR EL CENTRO DE LA IMAGEN
    #x positions
    x1=int(wid/8)#1/4 de la imagen en x
    x2=int(x1*5)#3/4 de la imagen en x
    #y positions 
    y1=int(he/8)#1/4 de la imagen  en y
    y2=int(he)#3/4 de la imagen en y
    #texto de procesar documento 
    cv2.rectangle(frame,(x1,y1+420),(400,525),(0,0,0),cv2.FILLED)
    cv2.putText(frame,"Procesando documento",(x1,y2+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    #Ubicamos el rectangulo en las zonas extraidas
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    #Realizamos el corte en el area delimitada
    recorte=frame[y1:y2,x1:x2]
    #preprocesamientos de la zona de interes
    nB=np.matrix(recorte[:,:,0])
    nG=np.matrix(recorte[:,:,1])
    nR=np.matrix(recorte[:,:,2])
    #Color
    Color=cv2.absdiff(nR,nG)
    
    #Binarizamos la imagen
    _,umbral=cv2.threshold(Color,40,255,cv2.THRESH_BINARY)
    #Extraemos los contornos  de la zona seleccionada
    contornos,_=cv2.findContours(umbral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #Primero los ordenamos del mas grande al mas pequeño
    contornos=sorted(contornos,key=lambda x:cv2.contourArea(x),reverse=True)
    #dibujamos los contornos extraidos
    for contorno  in contornos:
        area=cv2.contourArea(contorno)
        if area>500 and area<5000:
            #detectamos el documento
            x,y,ancho,alto=cv2.boundingRect(contorno)
            #extraemos las coordenadas
            xpi=x+x1#coordenada del documento  en x inicial
            ypi=y+y1#coordenada del documento  en y inicial

            xpf=x+ancho+x1 #coordenada en x final del documento
            ypf=y+alto+y1  #coordenada en y final del documento
            cv2.rectangle(frame,(xpi,ypi),(xpf,ypf),(255,255,0),2)
            #extraemos pixeles
            documento=frame[ypi:ypf,xpi:xpf]
            #extraemos el ancho y alto de fotogramas
            alp,anp,cp=documento.shape
            #procesamos los pixeles para enxtraer los valores de las documentos
            Mva=np.zeros((alp,anp))
            #normalizamos las matrices
            nBp=np.matrix(documento[:,:,0])
            nGp=np.matrix(documento[:,:,1])
            nRp=np.matrix(documento[:,:,2])
            #CREAMOS UNA MASCARA
            for col in range(0,alp):
                for fil in range(0,anp):
                    Max=max(nRp[col,fil],nGp[col,fil],nBp[col,fil])
                    Mva[col,fil]=255-Max
            #Binarizamos la imagen
            _,bin=cv2.threshold(Mva,150,255,cv2.THRESH_BINARY)
            #convertimos la matriz en una imagen
            bin=bin.reshape(alp,anp)
            bin=Image.fromarray(bin)
            bin=bin.convert("L")
            
            #nos aseguramos de tener un buen tamaño de documento
            if alp>=0 and anp>=0:
                t=83
                if t == 83 or t == 115:
                    txt=texto(bin)
                
                    #if para no mostrar basura
                    if len(txt)>=7:
                        print(txt)
                        CTexto=txt
#==========================================================================================

            break
    cv2.imshow("Ingresar texto",frame)
    #si es esc cierra
    t=cv2.waitKey(1)
    if(t==27):
        break

cap.release() 
cv2.destroyAllWindows()



