import os 
import cv2






def preprocess(image_name, image_dir, label_dir):
    image_path = os.path.sep.join([image_dir, image_name])
    name = image_name.split(".")[0]
    label_name = name + ".txt"
    label_path = os.path.sep.join([label_dir, label_name])

    return image_path, label_path, label_name


def draw_rect(img, bboxes, color=(255, 0, 0)):
    img = img.copy()
    height, width = img.shape[:2]
    for bbox in bboxes:
        center_x, center_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x = int((center_x - w / 2) * width)
        w = int(w * width)
        y = int((center_y - h / 2) * height)
        h = int(h * height)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    return img


def read_img(image_path, cvt_color=True):
    img = cv2.imread(image_path)
    if cvt_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_img(image, save_path, jpg_quality=None):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if jpg_quality:
        cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(save_path, image)


def read_label(label_path):
    with open(label_path) as f:
        conts = f.readlines()

    bboxes = []
    class_labels = []
    for cont in conts:
        cont = cont.strip().split()
        center_x, center_y, w, h = (
            float(cont[1]),
            float(cont[2]),
            float(cont[3]),
            float(cont[4]),
        )
        bboxes.append([center_x, center_y, w, h])
        class_labels.append(cont[0])
    return (bboxes, class_labels)


def display_img(image_path, label_path):
    img = read_img(image_path, cvt_color=False)
    bboxes = read_label(label_path)[0]
    img = draw_rect(img, bboxes)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def save_label(bboxes, class_labels, label_path):
    tem_lst = []
    for i, bbox in enumerate(bboxes):
        tem_lst.append(
            class_labels[i]
            + " "
            + str(bbox[0])
            + " "
            + str(bbox[1])
            + " "
            + str(bbox[2])
            + " "
            + str(bbox[3])
            + "\n"
        )

    with open(label_path, "w") as f:
        f.writelines(tem_lst)
