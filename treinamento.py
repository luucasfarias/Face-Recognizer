import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


# Percorrer todas as imagens que estão coletadas no diretorio pictures
def getImages():
    caminho = [os.path.join('pictures', foto) for foto in os.listdir('pictures')]
    print(caminho)

    faces = []
    ids = []

    for pathImagem in caminho:
        # leitura da imagem no diretorio e poe em escala de cinza
        imagemFace = cv2.cvtColor(cv2.imread(pathImagem), cv2.COLOR_BGR2GRAY)
        # pegando o id
        id = int(os.path.split(pathImagem)[-1].split('')[1])
        print(id)
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces

ids, faces = getImages()


# Inicando treinamento
print("Iniciando treinamento...")

eigenface.train(faces, ids)
eigenface.write(classifierEigen.yml)

fisherface.train(faces, ids)
fisher.write(classifierFisher.yml)

lbph.train(faces, ids)
lbph.write(classifierLBPH.yml)

print("Fim treinamento")
