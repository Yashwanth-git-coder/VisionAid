# import cv2
# import numpy as np
# from gtts import gTTS
# import pygame
# import os
# import speech_recognition as sr
# import pyttsx3
# import openai
#
# # Initialize pygame
# pygame.init()
#
# # Initialize the display (even though we won't use it)
# pygame.display.set_mode((1, 1))
#
# # Initialize pygame mixer for audio
# pygame.mixer.init()
#
# # Load YOLOv3 model and classes
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")
#
# # Initialize the webcam
# cap = cv2.VideoCapture(0)
#
# # Initialize speech recognition
# r = sr.Recognizer()
#
# # Initialize OpenAI API key
# openAPI_KEY = "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"
# openai.api_key = openAPI_KEY
#
#
# def SpeechText(command):
#     engine = pyttsx3.init()
#     engine.say(command)
#     engine.runAndWait()
#
#
# def record_text():
#     while True:
#         try:
#             with sr.Microphone() as source2:
#                 r.adjust_for_ambient_noise(source2, duration=0.2)
#                 print("I'm Listening")
#                 audio2 = r.listen(source2)
#                 MyText = r.recognize_google(audio2)
#                 print(MyText)
#                 return MyText
#
#         except sr.RequestError as e:
#             print("Couldn't find the required result; {0}".format(e))
#
#         except sr.UnknownValueError:
#             print("unknown value error")
#
#
# def send_to_chatGPT(messages, model="gpt-3.5-turbo"):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         max_tokens=100,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )
#
#     print(response.choices[0].message.content)
#
#     message = response.choices[0].message.content
#     messages.append(response.choices[0].message)
#
#     return message
#
#
# # Initialize chatbot messages
# messages = [{"role": "user", "content": "Please act as my personal assistant device and do not break character"}]
# calling = True
#
# while calling:
#     # Object detection code
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(net.getUnconnectedOutLayersNames())
#
#     detected_object = None
#
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#
#             if confidence > 0.5 and classes[class_id] in ["cell phone"]:
#                 detected_object = classes[class_id]
#
#     if detected_object:
#         print(f"Detected: {detected_object}")
#         tts = gTTS(text=f"Detected: {detected_object}", lang="en")
#         audio_file = "output.mp3"
#         tts.save(audio_file)
#
#         pygame.mixer.music.load(audio_file)
#         pygame.mixer.music.play()
#         pygame.mixer.music.set_endevent(pygame.USEREVENT)
#         pygame.event.wait()
#
#         # Generate information about the detected object using OpenAI
#         response = send_to_chatGPT(messages)
#         SpeechText(response)
#
#     cv2.imshow("Object Detection", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()

#--------------------------------------------part :- 02 -------------------------------------------------#

# import cv2
# import numpy as np
# from gtts import gTTS
# import pygame
# import os
# import speech_recognition as sr
# import pyttsx3
# import openai
#
# # Initialize pygame
# pygame.init()
#
# # Initialize the display (even though we won't use it)
# pygame.display.set_mode((1, 1))
#
# # Initialize pygame mixer for audio
# pygame.mixer.init()
#
# # Load YOLOv3 model and classes
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")
#
# # Initialize the webcam
# cap = cv2.VideoCapture(0)
#
# # Initialize speech recognition
# r = sr.Recognizer()
#
# # Initialize OpenAI API key
# openAPI_KEY = "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"
# openai.api_key = openAPI_KEY
#
#
# def SpeechText(command):
#     engine = pyttsx3.init()
#     engine.say(command)
#     engine.runAndWait()
#
#
# def record_text():
#     while True:
#         try:
#             with sr.Microphone() as source2:
#                 r.adjust_for_ambient_noise(source2, duration=0.2)
#                 print("I'm Listening")
#                 audio2 = r.listen(source2)
#                 MyText = r.recognize_google(audio2)
#                 print(MyText)
#                 return MyText
#
#         except sr.RequestError as e:
#             print("Couldn't find the required result; {0}".format(e))
#
#         except sr.UnknownValueError:
#             print("unknown value error")
#
#
# def send_to_chatGPT(messages, model="gpt-3.5-turbo"):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         max_tokens=100,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )
#
#     print(response.choices[0].message.content)
#
#     message = response.choices[0].message.content
#     messages.append(response.choices[0].message)
#
#     return message
#
#
# # Initialize chatbot messages
# messages = [{"role": "user", "content": "Please act as my personal assistant device and do not break character"}]
# calling = True
#
# while calling:
#     # Object detection code
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(net.getUnconnectedOutLayersNames())
#
#     detected_object = None
#
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#
#             if confidence > 0.5 and classes[class_id] in ["cell phone"]:
#                 detected_object = classes[class_id]
#
#     if detected_object:
#         print(f"Detected: {detected_object}")
#         tts = gTTS(text=f"Detected: {detected_object}", lang="en")
#         audio_file = "output.mp3"
#         tts.save(audio_file)
#
#         pygame.mixer.music.load(audio_file)
#         pygame.mixer.music.play()
#         pygame.mixer.music.set_endevent(pygame.USEREVENT)
#         pygame.event.wait()
#
#         # Generate a short description of the detected object using OpenAI
#         response = send_to_chatGPT(messages)
#         tts_description = gTTS(text=response, lang="en")
#         description_audio_file = "description_output.mp3"
#         tts_description.save(description_audio_file)
#
#         pygame.mixer.music.load(description_audio_file)
#         pygame.mixer.music.play()
#         pygame.mixer.music.set_endevent(pygame.USEREVENT)
#         pygame.event.wait()
#
#     cv2.imshow("Object Detection", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()

#--------------------------------------------part :-03---------------------------------------------------#

# import cv2
# import numpy as np
# from gtts import gTTS
# import pygame
# import os
# import speech_recognition as sr
# import pyttsx3
# import openai
#
# # Initialize pygame
# pygame.init()
#
# # Initialize the display (even though we won't use it)
# pygame.display.set_mode((1, 1))
#
# # Initialize pygame mixer for audio
# pygame.mixer.init()
#
# # Load YOLOv3 model and classes
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")
#
# # Initialize the webcam
# cap = cv2.VideoCapture(0)
#
# # Initialize speech recognition
# r = sr.Recognizer()
#
# # Initialize OpenAI API key
# openAPI_KEY = "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"
# openai.api_key = openAPI_KEY
#
#
# def SpeechText(command):
#     engine = pyttsx3.init()
#     engine.say(command)
#     engine.runAndWait()
#
#
# def record_text():
#     while True:
#         try:
#             with sr.Microphone() as source2:
#                 r.adjust_for_ambient_noise(source2, duration=0.2)
#                 print("I'm Listening")
#                 audio2 = r.listen(source2)
#                 MyText = r.recognize_google(audio2)
#                 print(MyText)
#                 return MyText
#
#         except sr.RequestError as e:
#             print("Couldn't find the required result; {0}".format(e))
#
#         except sr.UnknownValueError:
#             print("unknown value error")
#
#
# def send_to_chatGPT(messages, model="gpt-3.5-turbo"):
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         max_tokens=100,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )
#
#     print(response.choices[0].message.content)
#
#     message = response.choices[0].message.content
#     messages.append(response.choices[0].message)
#
#     return message
#
#
# # Initialize chatbot messages
# messages = [{"role": "system", "content": "You are now in assistant mode."}]
# calling = True
#
# while calling:
#     # Object detection code
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(net.getUnconnectedOutLayersNames())
#
#     detected_object = None
#
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#
#             if confidence > 0.5 and classes[class_id] in ["toothbrush"]:
#                 detected_object = classes[class_id]
#
#     if detected_object:
#         print(f"Detected: {detected_object}")
#         messages = [{"role": "user", "content": f"{detected_object}"}]  # Create a message with detected object
#         tts = gTTS(text=f"Detected: {detected_object}", lang="en")
#         audio_file = "output.mp3"
#         tts.save(audio_file)
#
#         pygame.mixer.music.load(audio_file)
#         pygame.mixer.music.play()
#         pygame.mixer.music.set_endevent(pygame.USEREVENT)
#         pygame.event.wait()
#
#         # Generate a one-line description of the detected object using OpenAI
#         response = send_to_chatGPT(messages)
#         tts_description = gTTS(text=response, lang="en")
#         description_audio_file = "description_output.mp3"
#         tts_description.save(description_audio_file)
#
#         pygame.mixer.music.load(description_audio_file)
#         pygame.mixer.music.play()
#         pygame.mixer.music.set_endevent(pygame.USEREVENT)
#         pygame.event.wait()
#
#     cv2.imshow("Object Detection", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()

#-------------------------------------------------------part :04-----------------------------------------#

import cv2
import numpy as np
from gtts import gTTS
import pygame
import os
import speech_recognition as sr
import pyttsx3
import openai

# Initialize pygame
pygame.init()

# Initialize the display (even though we won't use it)
pygame.display.set_mode((1, 1))

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load YOLOv3 model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize speech recognition
r = sr.Recognizer()

# Initialize OpenAI API key
openAPI_KEY = "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"
openai.api_key = openAPI_KEY


def SpeechText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


def record_text():
    while True:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print("I'm Listening")
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                print(MyText)
                return MyText

        except sr.RequestError as e:
            print("Couldn't find the required result; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown value error")


def send_to_chatGPT(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    print(response.choices[0].message.content)

    message = response.choices[0].message.content
    messages.append(response.choices[0].message)

    return message


# Initialize chatbot messages
messages = [{"role": "system", "content": "You are now in assistant mode."}]
calling = True

while calling:
    # Object detection code
    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    detected_object = None

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in ["cell phone", "toothbrush", "bottle", "wine glass",
                                                          "cup", "fork", "knife", "spoon", "bowl", "banana",
                                                          "apple", "orange", "mouse", "remote", "keyboard", "book"]:
                detected_object = classes[class_id]

    if detected_object:
        print(f"Detected: {detected_object}")
        messages = [{"role": "user", "content": f"{detected_object}"}]  # Create a message with detected object
        tts = gTTS(text=f"Detected: {detected_object}", lang="en")
        audio_file = "output.mp3"
        tts.save(audio_file)

        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        pygame.event.wait()

        # Generate a one-line description of the detected object using OpenAI
        response = send_to_chatGPT(messages)
        tts_description = gTTS(text=response, lang="en")
        description_audio_file = "description_output.mp3"
        tts_description.save(description_audio_file)

        pygame.mixer.music.load(description_audio_file)
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        pygame.event.wait()

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
