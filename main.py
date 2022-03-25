import os
from algos import *


def extract_letters(drawn_img, line, line_offset, save_path):
    letter_counter = 0
    line_h, line_w, _ = line.shape

    black_line = cv2.cvtColor(line, cv2.COLOR_RGB2GRAY)
    black_line = threshold(black_line, thr_value=64)
    contour_line, bboxes = contour(black_line)

    for x, y, width, height in bboxes:
        letter_occupancy = (width * height) / (line_h * line_w)
        '''
        if height < 15 or letter_occupancy < 0.00025:
            continue
        '''

        letter = line[y:y+height, x:x+width]
        cv2.imwrite(f"{save_path}/extracted_letters/{letter_counter}.jpg", letter)
        drawn_img = cv2.rectangle(drawn_img, (x + line_offset[0], y + line_offset[1]), (x+width + line_offset[0], y+height + line_offset[1]), (0, 0, 255), 1)

        cv2.imshow("line", drawn_img)
        cv2.waitKey(0)
        letter_counter += 1

    return drawn_img


def extract_lines(img, bboxes, save_path):
    line_counter = 0
    img_h, img_w, _ = img.shape
    drawn_img = img.copy()

    for x, y, width, height in bboxes:
        line_occupancy = (width * height) / (img_h * img_w)

        '''
        cv2.imshow("line", img[y:y+height, x:x+width])
        cv2.waitKey(0)
        print(height, line_occupancy)
        if height < 50 or line_occupancy < 0.001:
           continue
        '''

        line = img[y:y+height, x:x+width]
        drawn_img = extract_letters(drawn_img, line, (x, y), save_path)

        cv2.imwrite(f"{save_path}/extracted_lines/{line_counter}.jpg", line)
        drawn_img = cv2.rectangle(drawn_img, (x, y), (x+width, y+height), (0, 0, 255), 2)
        line_counter += 1

    return drawn_img


def extract_image(img, save_path):
    black_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    black_img = threshold(black_img, thr_value=64)
    cv2.imwrite(save_path+"/threshold.jpg", black_img)

    # black_img = erosion(black_img, kernel=(1, 2), num_iters=5)
    # cv2.imwrite(save_path + "/erosion.jpg", black_img)

    black_img = dilate(black_img, kernel=(50, 1), num_iters=4)
    cv2.imwrite(save_path+"/dilate.jpg", black_img)

    black_img, bboxes = contour(black_img, add_offset=False)
    cv2.imwrite(save_path+"/contour.jpg", black_img)

    drawn_img = extract_lines(img, bboxes, save_path=save_path)
    cv2.imwrite(save_path+"/final.jpg", drawn_img)


if __name__ == "__main__":
    for image_path in os.listdir("input_images/"):
        img = cv2.imread("input_images/"+image_path)
        img_name = image_path.split('/')[-1].split('.')[0]

        os.makedirs(f'output_images/{img_name}', exist_ok=True)
        os.makedirs(f'output_images/{img_name}/extracted_lines', exist_ok=True)
        os.makedirs(f'output_images/{img_name}/extracted_letters', exist_ok=True)

        extract_image(img, f'output_images/{img_name}')
