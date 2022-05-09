import cv2
import numpy as np
import imageio

print("READING VIDEO")
#vidcap = cv2.VideoCapture('media/among-us-dance-dance-BODY-LAYER.mp4')
vidcap = cv2.VideoCapture('media/among-us-dance-dance.mp4')
count = 0
frames = []
while True:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  if not success:
    break
  count += 1
  frames += [image]

print("Read %d", count)

print("Saving GIF file")
with imageio.get_writer("export.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

print("END")