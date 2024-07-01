import os
import re
import cv2
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageSequence
from predictor import Predictor
from config import model_path_prefix
import pdf2image
from pdf2image import convert_from_path, convert_from_bytes

# sudo docker run -t -d -p 8501:8501 fidelity
# sudo docker build -t fidelity .

# all_keywords = None
warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOAD_DIR = os.path.join(BASE_DIR, 'dataset', 'uploads')

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


def parse_args():
    args = argparse.Namespace(
        model_path_prefix=model_path_prefix,
        batch_size=2,
        max_seq_length=512,
        task_type="mrc",
        lang="en",
        device="gpu"
    )
    return args


@st.cache_data
def process_uploaded_file(_tif_images, _predictor, all_keywords, doctype = 'pdf'):
    OUTPUT_DIR = os.path.join(UPLOAD_DIR, str(time.time()))
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    result = dict.fromkeys(all_keywords, {'answer': [''], 'confidence_score': [0]})
    image_paths = list()
    ocr_results = dict()
    if doctype == "pdf":
        for i, image in enumerate(_tif_images):
            image_path = os.path.join(OUTPUT_DIR, f'page{i}.png')
            image_paths.append(image_path)
            image.save(image_path)
            
            docs = [image_path]

            outputs, ocr_result = _predictor.predict(docs)
            ocr_results[f'page{i}'] = ocr_result

            result_dict = {}
            for output in outputs[0]['result']:
                question = output['question']
                answer = output['answer']
                confidence_score = output['confidence_score']

                if question in result_dict.keys():
                    result_dict[question]['answer'].extend(answer)
                    result_dict[question]['confidence_score'].extend(confidence_score)
                    result_dict[question]['page'] = f'page{i}'
                else:
                    result_dict[question] = {
                        'answer': answer,
                        'confidence_score': confidence_score,
                        'page': f'page{i}'
                    }

            for key, value in result_dict.items():
                if result_dict[key]['confidence_score'] > result[key]['confidence_score']:
                    result[key] = value
    else:
        for i, image in enumerate(ImageSequence.Iterator(_tif_images)):
            image_path = os.path.join(OUTPUT_DIR, f'page{i}.png')
            image_paths.append(image_path)
            image.save(image_path)
            
            docs = [image_path]

            outputs, ocr_result = _predictor.predict(docs)
            ocr_results[f'page{i}'] = ocr_result

            result_dict = {}
            for output in outputs[0]['result']:
                question = output['question']
                answer = output['answer']
                confidence_score = output['confidence_score']

                if question in result_dict.keys():
                    result_dict[question]['answer'].extend(answer)
                    result_dict[question]['confidence_score'].extend(confidence_score)
                    result_dict[question]['page'] = f'page{i}'
                else:
                    result_dict[question] = {
                        'answer': answer,
                        'confidence_score': confidence_score,
                        'page': f'page{i}'
                    }

            for key, value in result_dict.items():
                if result_dict[key]['confidence_score'] > result[key]['confidence_score']:
                    result[key] = value
    return result, ocr_results, image_paths


def get_bboxes_images(result, all_ocr_results, image_paths):
    try:
        base_path = os.path.dirname(image_paths[0])
        
        pagewise_keys = dict()
        for key, value in result.items():
            # print("VALUE", value)
            if not value['page'] in pagewise_keys:
                pagewise_keys[value['page']] = [key]
            else:
                pagewise_keys[value['page']].append(key)
        
        images = list()
        for key, value in pagewise_keys.items():
            ocr_results = all_ocr_results[key]
            base_image = os.path.join(base_path, f"{key}.png")

            image = cv2.imread(base_image)
            for question in value:
                answer = re.sub(r'\s+', ' ', result[question]['answer'][0])
                answer = re.sub(r' , ', ', ', answer)
                answer = re.sub(r' - ', '-', answer)
                answer = re.sub(r" / ", "/", answer)
                for line in ocr_results:
                    if re.search(answer, re.sub(r'\s+', ' ', line[-1][0])):
                        bounding_box = np.array(line[0], dtype=np.int32)
                        cv2.polylines(image, [bounding_box], isClosed=True, color=(0, 255, 0), thickness=2)
                        # for box in line:
                        #     x1, y1 = min(box[0]), min(box[1])
                        #     x2, y2 = max(box[0]), max(box[1])
                        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        break
                cv2.imwrite(base_image, image)
            images.append(base_image)
        return images
    except KeyError:
        st.sidebar.write("Something went wrong, Please try again!")


args = parse_args()

def main():
    st.title("Fusemachines Information Extraction")

    # Sidebar content
    # st.sidebar.title("Upload and Process")
    uploaded_file = st.sidebar.file_uploader("Upload a file", ['jpg', 'png', 'jpeg', 'pdf', 'tif'])
    all_keywords = st.sidebar.text_input("Please ask you questions:")
    prev_uploaded_file_name = st.session_state.get("uploaded_file_name", None)
    add_question = st.sidebar.button("Add Question")
    process =st.sidebar.button("Search for Results")

    show_bboxes_images = False
    # Setting questions to empty list (if no questions are avaliable)
    if not uploaded_file:
        st.session_state.questions = []
    if uploaded_file is not None:
        if prev_uploaded_file_name != uploaded_file.name:
            # New file uploaded, update the session state with the new file name
            st.session_state.uploaded_file_name = uploaded_file.name
            prev_uploaded_file_name = uploaded_file.name
            st.cache_data.clear()
            st.success(f"New file '{uploaded_file.name}' uploaded.")
        doctype = 'pdf'
        if not show_bboxes_images and ("TIF" in uploaded_file.name or 'tif' in uploaded_file.name):
            tif_images = Image.open(uploaded_file)
            # tif_images = [im.convert('RGB') for idx, im in ImageSequence.Iterator(tif_images_)]
            # print("TYPE",type(tif_images), tif_images )
            num_frames = tif_images.n_frames
            frame_num = st.slider("Select Page", min_value=0, max_value=num_frames - 1, value=0)
            tif_images.seek(frame_num)
            st.image(tif_images, caption=f"Page {frame_num + 1}", use_column_width=True)
            doctype = 'tif'
        elif not show_bboxes_images and uploaded_file.type == "application/pdf":
            tif_images_pdf = pdf2image.convert_from_bytes(uploaded_file.read())
            # tif_images = tif_images.convert('RGB')
            tif_images = [im.convert('RGB') for im in tif_images_pdf]
            num_frames = len(tif_images)
            # print("Num Frames PDF",num_frames)
            frame_num = st.slider("Select Page", min_value=0, max_value=num_frames-1, value=0)
            # tif_images.seek(frame_num)
            st.image(tif_images[frame_num], caption=f"Page {frame_num + 1}", use_column_width=True)
            doctype = 'pdf'
        elif not show_bboxes_images:
            tif_images = [Image.open(uploaded_file).convert('RGB'),]
            st.image(tif_images, caption=f"Page", use_column_width=True)
            doctype = 'pdf'

        if add_question:
            st.session_state.questions.append(all_keywords)
            st.sidebar.write(st.session_state.questions)
        if len(all_keywords)>2 and process:
            if type(all_keywords)==str and len(st.session_state.questions)<=1:
                questions = [all_keywords,] 
            args.questions = [st.session_state.questions]
            predictor = Predictor(args)
            result, ocr_results, image_paths = process_uploaded_file(tif_images, predictor, all_keywords=st.session_state.questions, doctype=doctype)
        # breakpoint()
            images = get_bboxes_images(result, ocr_results, image_paths)

            st.header('Processed File')
            try:
                num_frames = len(images)
                if num_frames >1:
                    frame_num = st.slider("Select Image", min_value=0, max_value=num_frames - 1, value=0)
                    image = Image.open(images[frame_num])
                    st.image(image, caption=f"Page {frame_num + 1}", use_column_width=True)
                    show_bboxes_images = True
                    st.sidebar.write("Here are the Results:")
                else:
                    image = Image.open(images[0])
                    st.image(image, use_column_width=True)
                    show_bboxes_images = True
                    st.sidebar.write("Here are the Results:")
                table_data = list()
                for key, value in result.items():
                    table_data.append([key, value['answer'][0], round(value['confidence_score'][0], 4)])
                st.sidebar.table(pd.DataFrame(table_data, columns=['Keywords', 'Values', 'Score']))
                reset = st.sidebar.button("RESET")
                if reset:
                    st.session_state.questions = []
            except:
                pass
            
        # huggingface_model_button = st.sidebar.button("Compare with HuggingFace's Model")



if __name__ == "__main__":
    main()
