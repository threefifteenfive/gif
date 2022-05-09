import cv2
import numpy as np
import imageio
import argparse
import os


EXPORT_FILE = "export.gif"
IMG = None


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="file path to the desired input image")
    args = parser.parse_args()
    p = args.input_file
    file_path = os.getcwd() + "/" + p
    path = os.path.normpath(file_path)
    print("Input File Path: ", path)

    # Using cv2.imread() method
    img = cv2.imread(path)
    IMG = img


"""
Read the main video to generate the masks
"""
def read_video():
    """
    vidcap = cv2.VideoCapture('media/among-us-dance-dance-BODY-LAYER.mp4')
    count = 0
    body = []
    while True:
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        if not success:
            break
        count += 1
        body += [image]

    body_frames, position_list_UNUSED = greenscreen(body, False)
    """
    vidcap = cv2.VideoCapture('media/among-us-dance-dance-MASK_LAYER.mp4')
    count = 0
    mask = []
    while True:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        if not success:
            break
        count += 1
        mask += [image]
        #cv2.imshow('image', image)
        #cv2.waitKey(0)

    mask_frames, position_list = greenscreen(mask, True)

    for i in range(len(mask_frames)):
        mask = mask_frames[i]
        x, y, w, h = position_list[i]
        #cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), -1)
        #cv2.circle(mask, (x + w//2, y + h//2), radius=5, color=(0, 0, 255), thickness=-1)
        #cv2.imshow("centered", mask)
        #cv2.waitKey(0)
        #cv2.destroyWindow("centered")
        if IMG is None:
            print("NO INPUT IMG")
            continue
        print("mask_dim: ", (w, h))
        orig_dim = IMG.shape
        print("orig_dim: ", orig_dim)
        orig_w, orig_h = orig_dim
        w_scale = round((orig_w / w) * 100)
        h_scale = round((orig_h / h) * 100)
        print(w_scale, h_scale)





"""
Greenscreens the list of frames
green pixel value = 00FF00
"""
def greenscreen(frame_list, find_position):
    return_list = []
    position_list = []
    for img in frame_list:
        # convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # make mask of all green pixels
        key_mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
        # select inverse of that, non-green pixels
        key_mask = cv2.bitwise_not(key_mask)
        if find_position:
            position_list += [average_mask_pixels(key_mask)]
        # mask with original frame to remove the green pixels
        target = cv2.bitwise_and(img, img, mask=key_mask)
        return_list += [target]
    return return_list, position_list


"""
Find the average location of the mask pixels
"""
def average_mask_pixels(mask):
    #cv2.imshow("mask", mask)
    #cv2.waitKey(0)
    #cv2.destroyWindow("mask")
    ret, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow('binary', binary)
    #cv2.waitKey(0)
    #cv2.destroyWindow('binary')
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours:" + str(len(contours)))
    x, y, w, h = cv2.boundingRect(contours[-1])
    cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 3)
    #cv2.imshow("result", mask)
    #cv2.waitKey(0)
    #cv2.destroyWindow("result")
    return (x, y, w, h)


"""
Write the list of frames (frames) to output gif
"""
def export(frames):
    print("Saving GIF file")
    with imageio.get_writer(EXPORT_FILE, mode="I") as writer:
        for idx, frame in enumerate(frames):
            print("Adding frame to GIF file: ", idx + 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
    print("END")


if __name__ == "__main__":
    generate()
    read_video()
