import cv2

# Definindo o classificador
classificador = cv2.CascadeClassifier("classifiers/haarcascade-frontalface-default.xml")


# Aciona a webcam
camera = cv2.VideoCapture()

# A linha 8 faz a leitura da camera
# A linha 9 exibe a imagem capturada que foi armazenada na variavel imagem
while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # transformando p/ escala de cinza
    # Detecta imagem em escala cinza
    facesDetectadas = classificador.detectMultiScale(
        imagemCinza, scaleFactor=1.5, minSize=(100,100))

    # Desenhar o retângulo da imagem detectada
    # l = largura, a = altura
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 3)

    cv2.imshow("Face", imagem)
    cv2.waitkey(1)


# A linha xx libera a memoria e a linha yy destroi a janela de exibição
camera.release()
cv2.destroyAllWindows()