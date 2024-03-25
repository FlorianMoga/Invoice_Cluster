import cv2
import numpy as np
from PIL import Image
import os

CLASSES_FOLDER = r"InvoiceCluster\flask-app\static\clustering\classes"
CLUSTERS_FOLDER = r"InvoiceCluster\flask-app\static\clustering\clusters"
EMBEDDINGS = r"InvoiceCluster\flask-app\static\clustering\test.npy"


def detect_logo(model, img_path, mode):
    if type(img_path) == str:
        orig_x = cv2.imread(img_path)
    elif type(img_path) == np.ndarray:
        orig_x = img_path
    else:
        raise TypeError("Filetype not supported")

    yolov8_output = model(orig_x)

    # no logo found
    if len(yolov8_output[0].boxes.data) == 0:
        return "no", orig_x
    else:
        xywh = yolov8_output[0].boxes.xyxy.cpu().numpy().astype(int)
        xywh = xywh[0]
        x1, y1 = xywh[0], xywh[1]
        x2, y2 = xywh[2], xywh[3]

        if mode == "roi":
            logo_roi = orig_x.copy()
            logo_roi = logo_roi[y1:y2, x1:x2]
            return "roi", logo_roi
        elif mode == "bb":
            annotated = orig_x.copy()
            annotated = cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255),
                                      thickness=1, lineType=cv2.LINE_AA)

            return xywh, annotated


def extract_embeddings(model, transforms, img_path):
    if type(img_path) == str:
        logo_roi = cv2.imread(img_path)
    elif type(img_path) == np.ndarray:
        logo_roi = img_path
    else:
        raise TypeError("Filetype not supported")

    pil_logo = Image.fromarray(logo_roi)
    logo_t = transforms(pil_logo)
    logo_t = logo_t.unsqueeze(0)
    embedding = model(logo_t)
    embedding = embedding.detach().cpu().numpy()
    return embedding[0]


def get_mean(arr):
    return np.mean(arr, axis=0)


def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_ab = norm_a * norm_b
    if norm_ab < 1e-8:
        norm_ab = 1e-8
    return np.dot(a, b) / norm_ab


def load_embeddings():
    image_folder_files = os.listdir(CLASSES_FOLDER)
    image_folder_files = [int(img.split('.')[0]) for img in image_folder_files]
    image_folder_files.sort()

    f_dict = {}
    for file_class in image_folder_files:
        f_dict[file_class] = []

    with open(EMBEDDINGS, 'rb') as f:
        for key in f_dict:
            f_dict[key] = np.load(f)

    return f_dict


def save_embeddings(embeddings):
    embeddings = dict(sorted(embeddings.items()))
    with open(EMBEDDINGS, 'wb') as f:
        for key in embeddings:
            np.save(f, embeddings[key])


def cluster_embeddings(embedding, invoice, filename, logo_roi, thresh):
    embeddings = load_embeddings()
    class_num = len(embeddings.keys())

    biggest_sim = 0
    # iterate the present clusters to choose the best one
    for key in embeddings:
        cls_mean = get_mean(embeddings[key])

        # compare the sample with all clusters mean
        cls_sim = cosine_sim(embedding, cls_mean)

        # if bigger than the actual, update the biggest similarity and change the biggest similarity cluster
        if cls_sim > biggest_sim:
            biggest_sim = cls_sim
            best_cluster = int(key)

    # if the biggest similarity does not exceed the threshold, create new cluster with that sample
    if biggest_sim < thresh:
        new_dir = CLUSTERS_FOLDER + "\\" + str(class_num)
        os.mkdir(new_dir)
        cv2.imwrite(new_dir + "\\" + filename, invoice)

        cv2.imwrite(CLASSES_FOLDER + f"\\{class_num}.png", logo_roi)

        new_class = np.empty((0, 2208))
        embeddings[class_num] = np.append(new_class, np.array([embedding]), axis=0)

    # if the biggest similarity exceeds the threshold, add it to the best cluster
    else:
        dir = CLUSTERS_FOLDER + "\\" + str(best_cluster)
        cv2.imwrite(dir + "\\" + filename, invoice)

        best_cluster_embeddings = embeddings[best_cluster]
        best_cluster_embeddings = np.append(best_cluster_embeddings, np.array([embedding]), axis=0)
        embeddings[best_cluster] = best_cluster_embeddings

    for cluster in embeddings:
        print(cluster, len(embeddings[cluster]))

    save_embeddings(embeddings)
