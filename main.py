import cv2

# Incarcam "cascada" (algoritmul pre-antrenat de OpenCV pentru fete)
# Acesta vine gata instalat cu OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Pornim camera (0 este de obicei camera de laptop)
video_capture = cv2.VideoCapture(0)

print("Camera a pornit! Apasa 'q' pentru a iesi.")

while True:
    # Citim cadrul de la camera
    ret, frame = video_capture.read()
    
    if not ret:
        print("Eroare la citirea camerei.")
        break

    # OpenCV lucreaza mai bine cu imagini alb-negru pentru detectie
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectam fetele
    # scaleFactor=1.1 inseamna ca micsoram imaginea cu 10% la fiecare pas pentru a gasi fete de marimi diferite
    # minNeighbors=5 ajuta la eliminarea erorilor (fete false)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenam un patrat pentru fiecare fata gasita
    for (x, y, w, h) in faces:
        # Desenam dreptunghiul verde (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Putem scrie un text generic
        cv2.putText(frame, "Persoana", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Afisam rezultatul
    cv2.imshow('Detectie Faciala Simpla', frame)

    # Iesim cu tasta 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Curatenie la final
video_capture.release()
cv2.destroyAllWindows()