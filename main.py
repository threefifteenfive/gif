import cv2
import numpy as np
import imageio
import argparse
import os


EXPORT_FILE = "export.gif"


def read_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="file path to the desired input image")
    args = parser.parse_args()
    p = args.input_file
    file_path = os.getcwd() + "/" + p
    path = os.path.normpath(file_path)
    print("Input File Path: ", path)

    # Using cv2.imread() method
    img = cv2.imread(path)
    #cv2.imshow("input image", img)
    #cv2.waitKey(0)
    #cv2.destroyWindow("input image")
    return img


"""
Read the main video to generate the masks
"""
def read_video(input_image):

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

    body_frames, position_list_UNUSED = greenscreen(body, False, [])

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

    binary_mask_list = []
    mask_frames, position_list = greenscreen(mask, True, binary_mask_list)

    final_frames = []

    for i in range(len(mask_frames)):
        mask = mask_frames[i]
        x, y, w, h = position_list[i]
        #cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 3)
        #cv2.circle(mask, (x + w//2, y + h//2), radius=5, color=(0, 0, 255), thickness=-1)
        #cv2.imshow("centered", mask)
        #cv2.waitKey(0)
        #cv2.destroyWindow("centered")
        if input_image is None:
            print("NO INPUT IMG")
            continue

        print("mask_dim: ", (w, h))
        print("orig_dim: ", input_img.shape)
        orig_w, orig_h = input_img.shape[0], input_image.shape[1]
        w_scale = round((orig_w / w), 2)
        h_scale = round((orig_h / h), 2)
        print(w_scale, h_scale)

        new_dim = (int(orig_w // w_scale), int(orig_h // w_scale))
        # since width is larger than height
        if new_dim[0] != w:
            resized_offset = w - new_dim[0]
            new_dim = (new_dim[0] + resized_offset, new_dim[1] + resized_offset)
        # resize image
        resized = cv2.resize(input_img, new_dim, interpolation=cv2.INTER_AREA)
        # create mask matrix
        blank_image = np.zeros((new_dim[1], new_dim[0], 3), np.uint8)
        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY);
        # mask out the part of the image to be used
        center_height = new_dim[1] // 2
        bot = int(center_height - h // 2)
        cv2.rectangle(gray, (0, bot), (w, bot + h), (255, 255, 255), -1)
        target = cv2.bitwise_and(resized, resized, mask=gray)
        print("new dimensions", new_dim)
        """
        cv2.imshow("resized", target)
        cv2.waitKey(0)
        cv2.destroyWindow("resized")
        """

        # need to align the section with the mask section in the matrix
        print("mask shape", mask.shape)
        blank = np.zeros_like(mask)
        #wall[x:x + block.shape[0], y:y + block.shape[1]] = block
        #blank[x:x + w, y:y + h] = target[0:w, bot:bot + h]
        blank[y:y + h, x:x + w] = target[bot:bot+h, 0:w]
        print("blank shape", blank.shape)
        #cv2.imshow("translated", blank)
        #cv2.waitKey(0)
        #cv2.destroyWindow("translated")

        # mask with the mask from greenscreen (need to save that from greenscreen method)
        """
        cv2.imshow("mask", binary_mask_list[i])
        cv2.waitKey(0)
        cv2.destroyWindow("mask")
        """
        input_masked_layer = cv2.bitwise_and(blank, blank, mask=binary_mask_list[i])
        """
        cv2.imshow("masked input layer", input_masked_layer)
        cv2.waitKey(0)
        cv2.destroyWindow("masked input layer")
        """
        # bitwise or them together
        """
        cv2.imshow("body layer", body_frames[i])
        cv2.waitKey(0)
        cv2.destroyWindow("body layer")
        """
        composit = np.zeros_like(mask)
        cv2.bitwise_or(body_frames[i], input_masked_layer, composit)
        """
        cv2.imshow("composited", composit)
        cv2.waitKey(0)
        cv2.destroyWindow("composited")
        """
        """
        cv2.imshow("mask part", mask_frames[i])
        cv2.waitKey(0)
        cv2.destroyWindow("mask part")
        """
        # Create the overlay
        alpha = 0.5
        # Change this into bool to use it as mask
        mask = binary_mask_list[i].astype(bool)
        composit[mask] = cv2.addWeighted(composit, 1 - alpha, mask_frames[i], alpha, 0)[mask]
        """
        cv2.imshow("alpha", composit)
        cv2.waitKey(0)
        cv2.destroyWindow("alpha")
        """
        final_frames += [composit]

    return final_frames







"""
Greenscreens the list of frames
green pixel value = 00FF00
"""
def greenscreen(frame_list, find_position, mask_list):
    return_list = []
    position_list = []
    for img in frame_list:
        # convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # make mask of all green pixels
        key_mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
        # select inverse of that, non-green pixels
        key_mask = cv2.bitwise_not(key_mask)
        """
        cv2.imshow("mask", key_mask)
        cv2.waitKey(0)
        cv2.destroyWindow("mask")
        """
        if find_position:
            position_list += [average_mask_pixels(key_mask)]
            mask_list += [key_mask]
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
    #cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0), 3)
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
    input_img = read_input()
    frame_list = read_video(input_img)
    export(frame_list)
