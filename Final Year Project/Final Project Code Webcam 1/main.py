import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import requests
import datetime
import telebot
import os
from telegram import Bot
from telegram.ext import Updater, MessageHandler, filters
import threading
from PIL import ImageGrab
import io

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']


def take_screenshot():
    screenshot = ImageGrab.grab()
    return screenshot


def take_screenshot_and_send():
    screenshot = take_screenshot()
    times = datetime.datetime.now()
    timeconvert = times.strftime("%m/%d/%Y ,%H: %M: %S")
    telegram_bot_sendtext_emerg1("Fallen user at: " + timeconvert, screenshot)


def telegram_bot_sendtext_main(bot_message):
    bot_token = '6173338669:AAEuFBA00dAaGY8WXoQL554kRghgrTHXqEE'
    bot_chatID = '979726749'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()


def handle_message(update, context):
    message = update.message.text.lower()  # convert message to lowercase for case-insensitive matching

    if message == "accept":
        context.bot.send_message(chat_id='979726749', text="Ok, Continue Monitoring")
    elif message == "deny":
        context.bot.send_message(chat_id='979726749', text="Ok, Sending alert to next contact")
        times = datetime.datetime.now()
        timeconvert = times.strftime("%m/%d/%Y ,%H: %M: %S")
        telegram_bot_sendtext_emerg1("Fallen user at: " + timeconvert)
    else:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that.")

    updater = Updater(token='6173338669:AAEuFBA00dAaGY8WXoQL554kRghgrTHXqEE', use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(MessageHandler(filters.text, handle_message))

    updater.start_polling()
    updater.idle()


def telegram_bot_sendtext_emerg1(bot_message, screenshot):
    bot_token = '6055310733:AAFXn2aDCU3VL9fVxygxIS1hNWRbe11VfJA'
    bot_chatID = '979726749'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendPhoto'

    # Convert the 'Image' object to bytes
    image_bytes = io.BytesIO()
    screenshot.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    response = requests.post(send_text, data={'chat_id': bot_chatID, 'caption': bot_message},
                             files={'photo': ('screenshot.png', image_bytes)})

    return response.json()


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


prev_angle = None

import time

cap = cv2.VideoCapture(1)


def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    global prev_angle
    fallen = False
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

        # Check if person has fallen
        left_ankle = person[15]
        right_ankle = person[16]
        left_eye = person[1]
        right_eye = person[2]
        left_shoulder = person[5]
        right_shoulder = person[6]
        left_hip = person[11]
        right_hip = person[12]

        if left_ankle[2] > confidence_threshold and right_ankle[2] > confidence_threshold \
                and left_eye[2] > confidence_threshold and right_eye[2] > confidence_threshold:

            # Compute distances
            ankle_distance = np.linalg.norm(left_ankle[:2] - right_ankle[:2])
            eye_distance = np.linalg.norm(left_eye[:2] - right_eye[:2])
            ankle_to_eye_distance = np.linalg.norm(
                (left_ankle[:2] + right_ankle[:2]) / 2 - (left_eye[:2] + right_eye[:2]) / 2)

            # Calculate angle between hips and shoulders
            if left_hip[2] > confidence_threshold and right_hip[2] > confidence_threshold \
                    and left_shoulder[2] > confidence_threshold and right_shoulder[2] > confidence_threshold:

                # Compute vector representing hips and shoulders
                hip_vector = right_hip[:2] - left_hip[:2]
                shoulder_vector = right_shoulder[:2] - left_shoulder[:2]

                # Compute dot product and angle
                dot_product = np.dot(hip_vector, shoulder_vector)
                hip_norm = np.linalg.norm(hip_vector)
                shoulder_norm = np.linalg.norm(shoulder_vector)
                angle = math.acos(dot_product / (hip_norm * shoulder_norm))

                # Check if there has been a drastic change in angle
                prev_angle = getattr(loop_through_people, 'prev_angle', None)
                if prev_angle is not None and abs(angle - prev_angle) > 0:
                    fallen = False
                    screenshot = False
                    cv2.putText(frame, " POTENTIAL FALL!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if prev_angle >= angle and not fallen:
                        fallen = True
                        cv2.putText(frame, " FALLEN!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Start a new thread for image capture and sending
                        t = threading.Thread(target=take_screenshot_and_send)
                        t.start()
                        i = 0
                        i = i + 1
                        screenshot = True
                        prev_angle = 0

                # Save current angle for next iteration
                loop_through_people.prev_angle = angle

    return fallen


while cap.isOpened():
    ret, frame = cap.read()

    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    EDGES = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }

    # Render keypoints
    loop_through_people(frame, keypoints_with_scores, EDGES, 0)

    cv2.imshow('Movenet Multipose', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
