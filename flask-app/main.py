from flask import flash, request, redirect, url_for, render_template, current_app, render_template_string
from werkzeug.utils import secure_filename
from app import app, MODEL, FEATURE_EXTRACTOR, TRANSFORMS_DENSENET, THRESHOLD
from utils.utils import *
import os
# import time


ALLOWED_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng'}
TEMPLATE = "index (2).html"
IMAGE_FOLDER = r"static/clustering/classes"
CLUSTERS_FOLDER = r"clustering/classes"
CLASSES_FOLDER = r"static/clustering/clusters/"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_logos():
    images = []
    for filename in os.listdir(IMAGE_FOLDER):
        cluster_number = filename.split(".")[0]
        title = len(os.listdir(CLASSES_FOLDER + cluster_number))
        path = CLUSTERS_FOLDER + "/" + filename
        if title == 1:
            sample = "sample"
        else:
            sample = "samples"
        images.append({'title': f"Cluster {cluster_number} : {title} {sample}", 'path': path})
    return images


@app.route('/render_template', methods=['GET'])
def render_template_route():
    images = load_logos()
    return render_template(TEMPLATE, images=images)


@app.route('/')
def upload_form():
    images = load_logos()
    return render_template(TEMPLATE, images=images)


@app.route('/invocations', methods=['POST'])
def upload_image():

    return_radio = request.form.getlist('return_type')
    return_radio = return_radio[0]

    # Process the uploaded files and save them
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files[]')

    output_filepaths = []
    images = load_logos()

    if 1 <= len(files) <= 7:
        for file in files:
            if file.filename == "":
                pass
            else:
                if file and allowed_file(file.filename):
                    input_filename = secure_filename(file.filename)
                    input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    file.save(input_filepath)

                    input_filepath_extension = input_filepath.split(".")[-1]

                    # Process the file and save the results
                    flag, logo_roi = detect_logo(MODEL, input_filepath, return_radio)
                    flag_type = type(flag)
                    if flag_type == str and flag == "no":
                        image_flag = "_nologo." + input_filepath_extension
                    elif flag_type == str and flag == "roi":
                        image_flag = "_roi." + input_filepath_extension
                    else:
                        extension = ["_"+str(coord) for coord in flag]
                        extension = "".join(extension)
                        extension = extension + "." + input_filepath_extension
                        image_flag = extension

                    output_filename = f"{'.'.join(input_filename.split('.')[:-1])}{image_flag}"
                    output_filepath = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
                    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

                    cv2.imwrite(output_filepath, logo_roi)
                    output_filepaths.append(output_filepath)
                else:
                    flash(f'Allowed image types are -> {ALLOWED_EXTENSIONS}')
                    return redirect(request.url)

        archive_name = None

        # Render the template with the updated filepaths
        return render_template(TEMPLATE, filepaths=output_filepaths, archive_name=archive_name, images=images)
    else:
        flash('Please select at least 1 and maximum 5 files')
        return render_template(TEMPLATE, images=images)


@app.route('/cluster/<file>', methods=['GET', 'POST'])
def cluster(file):
    result_path = os.path.join(current_app.root_path, app.config['RESULTS_FOLDER'], file)
    result_path_extension = result_path.split('.')[-1]

    # types of paths
    # \flask-app\static/outputs/eon_sample_roi.jpg
    # \flask-app\static/outputs/eon_sample_nologo.jpg
    # \flask-app\static/outputs/eon_sample_x1_y1_x2_y2.jpg

    split_path = file.split("_")
    split_path[-1] = split_path[-1][:-4]

    init_filepath = current_app.root_path + "/" + app.config['UPLOAD_FOLDER'] + "_".join(
        split_path[0:-1]) + '.' + result_path_extension

    if split_path[-1] == 'roi':
        image = cv2.imread(init_filepath)
        init_filename = "_".join(split_path[0:-1]) + '.' + result_path_extension

        logo_roi = cv2.imread(result_path)
        # start = time.time()
        embedding = extract_embeddings(FEATURE_EXTRACTOR, TRANSFORMS_DENSENET, logo_roi)
        cluster_embeddings(embedding, image, init_filename, logo_roi, THRESHOLD)
        # print(time.time()-start)

    elif split_path[-1] == 'nologo':
        image = cv2.imread(init_filepath)
        init_filename = "_".join(split_path[0:-1]) + '.' + result_path_extension
        cv2.imwrite(app.config['NO_LOGO_FOLDER'] + "\\" + init_filename, image)

    else:
        x1, y1, x2, y2 = int(split_path[-4]), int(split_path[-3]), int(split_path[-2]), int(split_path[-1])
        init_filepath = current_app.root_path + "/" + app.config['UPLOAD_FOLDER'] + "_".join(split_path[0:-4]) + '.' + result_path_extension
        init_filename = "_".join(split_path[0:-4]) + '.' + result_path_extension

        image = cv2.imread(init_filepath)
        logo_roi = image[y1:y2, x1:x2]
        # start = time.time()
        embedding = extract_embeddings(FEATURE_EXTRACTOR, TRANSFORMS_DENSENET, logo_roi)
        cluster_embeddings(embedding, image, init_filename, logo_roi, THRESHOLD)
        # print(time.time()-start)

    return ('', 204)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='outputs/' + filename), code=301)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
