import cv2
import numpy as np

# Fomato da nomenclatura da imagem: pessoa.{id}.{numero_img}.jpg

# Definindo o classificador
classificador = cv2.CascadeClassifier("classifiers/haarcascade-frontalface-default.xml")
classificadorOlhos = cv2.CascadeClassifier("classifiers/haarcascade-eye.xml")
# Aciona a webcam
camera = cv2.VideoCapture()
amostra = 1
totalAmostras = 30
id = input('Digitar ID: ')
largura, altura = 220, 220

# A linha 8 faz a leitura da camera
# A linha 9 exibe a imagem capturada que foi armazenada na variavel imagem
while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transformando p/ escala de cinza
    # Detecta imagem em escala cinza
    facesDetectadas = classificador.detectMultiScale(
        imagemCinza, scaleFactor=1.5, minSize=(150,150))

    # Desenhar o retângulo da imagem detectada
    # l = largura, a = altura
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 3)

        # detecta olhos em uma regiao da face
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho)
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol + oy + oa), (0, 255, 0), 3)

            if cv2.waitkey(1) & 0xFF  == ord('x'):
                '''
                    Checar a luminosidade como uma condição pra coletar imagem,
                    o valor de 110 é um valor padrão de luminosidade ideal, fica a
                    critério da necessidade o valor.
                '''
                if np.average(imagemCinza) > 110:
                    imgFace =  cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite("pictures/pessoa." + str(id) + "." + str(amostra) + ".jpg",
                        imgFace)
                    print("ok")
                    amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitkey(1)

    # Checa se o total de amostras estiver no limite e pausa a captura
    if(amostra  >= totalAmostras + 1):
        break

# A linha xx libera a memoria e a linha yy destroi a janela de exibição
print('chegar aqui é bom')
camera.release()
cv2.destroyAllWindows()