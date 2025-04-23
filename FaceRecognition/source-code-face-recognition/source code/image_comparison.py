import cv2
import face_recognition

img = cv2.imread(r"C:\Users\produ\Downloads\WhatsApp Image 2024-11-15 at 18.13.38_35cf4499.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread(r"C:\Users\produ\Downloads\WhatsApp Image 2024-11-15 at 18.13.38_f704a136.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.imshow("Img", img)
cv2.imshow("Img 2", img2)
cv2.waitKey(0)