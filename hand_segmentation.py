import os
import numpy as np
import cv2


if __name__ == "__main__":

    im_subdir = os.listdir("./dataset/")
    co = 0

    # iterate through all images in "./dataset" folder
    for subdir in im_subdir:
        # skip folders for which mask has already been created
        if subdir[-5:] == "_mask":
            continue
        if not os.path.exists(os.path.join('./dataset', subdir + "_mask")):
            os.mkdir(os.path.join('./dataset', subdir + "_mask"))
        else:
            continue
        all_folders = os.listdir(os.path.join('./dataset', subdir))
        for folder in all_folders:
            all_images = os.listdir(os.path.join('./dataset', subdir, folder))
            for image in all_images:

                if not os.path.exists(os.path.join('./dataset', subdir + "_mask", folder)):
                    os.mkdir(os.path.join('./dataset', subdir + "_mask", folder))

                # location of input image and save location of output image
                path_from = os.path.join('./dataset', subdir, folder, image)
                path_to = os.path.join('./dataset', subdir + "_mask", folder, image)

                # load input image
                img = cv2.imread(path_from)

                # skin color map in bgra
                bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                lower_mask = cv2.inRange(bgra, (20, 40, 95, 15), (255, 255, 255, 255))
                rg_mask = (bgra[:,:,2] > bgra[:,:,1]).astype(np.uint8) * 255
                rb_mask = (bgra[:,:,2] > bgra[:,:,0]).astype(np.uint8) * 255
                diff_rg_mask = ((bgra[:,:,2] - bgra[:,:,1]) > 15).astype(np.uint8) * 255
                bgra_res_mask1 = cv2.bitwise_and(lower_mask, rg_mask)
                bgra_res_mask2 = cv2.bitwise_and(rb_mask, diff_rg_mask)
                bgra_mask = cv2.bitwise_and(bgra_res_mask1, bgra_res_mask2)

                # skin color mask in hsv
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv_mask = cv2.inRange(hsv, (0, 0.23*255, 0), (50, 0.68*255, 255))

                # skin color mask in ycrcb
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                lower_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 255, 255))
                cr_mask1 = (ycrcb[:,:,1] <= (1.5862 * ycrcb[:,:,2] + 20)).astype(np.uint8) * 255
                cr_mask2 = (ycrcb[:,:,1] >= (0.3448 * ycrcb[:,:,2] + 76.2069)).astype(np.uint8) * 255
                cr_mask3 = (ycrcb[:,:,1] >= (-4.5652 * ycrcb[:,:,2] + 234.5652)).astype(np.uint8) * 255
                cr_mask4 = (ycrcb[:,:,1] <= (-1.15 * ycrcb[:,:,2] + 301.75)).astype(np.uint8) * 255
                cr_mask5 = (ycrcb[:,:,1] <= (-2.2857 * ycrcb[:,:,2] + 432.85)).astype(np.uint8) * 255
                ycrcb_res_mask1 = cv2.bitwise_and(lower_mask, cr_mask1)
                ycrcb_res_mask2 = cv2.bitwise_and(cr_mask2, cr_mask3)
                ycrcb_res_mask3 = cv2.bitwise_and(cr_mask4, cr_mask5)
                ycrcb_res_mask12 = cv2.bitwise_and(ycrcb_res_mask1, ycrcb_res_mask2)
                ycrcb_mask = cv2.bitwise_and(ycrcb_res_mask12, ycrcb_res_mask3)

                # final skin mask
                bgra_hsv_mask = cv2.bitwise_and(bgra_mask, hsv_mask)
                bgra_ycrcb_mask = cv2.bitwise_and(bgra_mask, ycrcb_mask)
                hand = cv2.bitwise_or(bgra_hsv_mask, bgra_ycrcb_mask)
                hand = cv2.medianBlur(hand, 3)
                hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

                # save output image
                cv2.imwrite(path_to, hand)
                print("Mask of {} save to {}.".format(image, path_to))
                co += 1

print("{} generated masks.".format(co))
