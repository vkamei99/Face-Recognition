import face_recognition
import cv2
import numpy as np

# Carregar a imagem de referência e aprender a reconhecer ela
imagem_referencia = face_recognition.load_image_file("referencia.jpg")
codificacao_referencia = face_recognition.face_encodings(imagem_referencia)[0]

# Iniciar a webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar um frame de vídeo
    ret, frame = video_capture.read()

    # Reduzir o tamanho do frame para processamento mais rápido
    menor_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converter a imagem BGR (OpenCV) para RGB (face_recognition)
    rgb_menor_frame = menor_frame[:, :, ::-1]

    # Encontrar todas as faces e codificações no frame atual
    localizacoes_faces = face_recognition.face_locations(rgb_menor_frame)
    codificacoes_faces = face_recognition.face_encodings(rgb_menor_frame, localizacoes_faces)

    # Loop por todas as faces detectadas e comparar com a face de referência
    for (top, right, bottom, left), face_encoding in zip(localizacoes_faces, codificacoes_faces):
        matches = face_recognition.compare_faces([codificacao_referencia], face_encoding)
        nome = "Desconhecido"

        if True in matches:
            nome = "Pessoa Conhecida"

        # Ajustar a posição das coordenadas da face para o frame original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenhar um retângulo ao redor da face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Colocar o nome abaixo do retângulo
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mostrar o resultado
    cv2.imshow('Video', frame)

    # Pressionar 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar as janelas
video_capture.release()
cv2.destroyAllWindows()
