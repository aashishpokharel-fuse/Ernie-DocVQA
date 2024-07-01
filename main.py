import os
import time
import argparse
import pandas as pd
import streamlit as st
from PIL import Image, ImageSequence
from predictor import Predictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOAD_DIR = os.path.join(BASE_DIR, 'dataset', 'uploads')

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


def parse_args():
    args = argparse.Namespace(
        model_path_prefix=st.sidebar.text_input("Model Path Prefix", value="model/fidelity-413/inference", help="Path prefix of the inference model."),
        batch_size=st.sidebar.number_input("Batch Size", value=4, help="Batch size per GPU for inference."),
        max_seq_length=st.sidebar.number_input("Max Sequence Length", value=512, help="The maximum input sequence length. Sequences longer than this will be split automatically."),
        task_type=st.sidebar.selectbox("Task Type", ["mrc", "cls", "ner"], help="Specify the task type."),
        lang=st.sidebar.selectbox("Language", ["en", "ch"], help="Specify the language."),
        device=st.sidebar.selectbox("Device", ["gpu", "cpu"], help="Select which device to use for model inference.")
    )
    return args


@st.cache(suppress_st_warning=True)
def process_uploaded_file(args, tif_images):
    OUTPUT_DIR = os.path.join(UPLOAD_DIR, str(time.time()))
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    result = dict()
    for i, image in enumerate(ImageSequence.Iterator(tif_images)):
        image_path = os.path.join(OUTPUT_DIR, f'page{i}.jpg')
        image.save(image_path)
        if args.task_type == "mrc":
            args.questions = [
                [
                    "Recording Company Name",
                    "Recording Doc Number",
                    "Recording Document Date",
                    "APN",
                    "Document Type",
                    "Map Book",
                    "Map Page",
                    "Lot No",
                    "Party Name"
                ],
            ]
            docs = [image_path]
        elif args.task_type == "cls":
            docs = [image]
        elif args.task_type == "ner":
            docs = [image]
        else:
            raise ValueError("Unsupported task type: {}".format(args.task_type))

        predictor = Predictor(args)
        outputs = predictor.predict(docs)

        result_dict = {}
        for output in outputs[0]['result']:
            question = output['question']
            answer = output['answer']
            confidence_score = output['confidence_score']

            if question in result_dict:
                result_dict[question]['answer'].extend(answer)
                result_dict[question]['confidence_score'].extend(confidence_score)
            else:
                result_dict[question] = {'answer': answer, 'confidence_score': confidence_score}
        if not result:
            result = result_dict
            continue

        for key, value in result_dict.items():
            if result_dict[key]['confidence_score'] > result[key]['confidence_score']:
                result[key] = value
    return result


def main():
    st.title("Fidelity Information Extraction")

    # Sidebar content
    st.sidebar.title("Upload and Process")
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["tif", "tiff", "jpg", "pdf"])

    if uploaded_file is not None:
        tif_images = Image.open(uploaded_file)
        num_frames = tif_images.n_frames
        frame_num = st.slider("Select Page", min_value=0, max_value=num_frames - 1, value=1)
        image_container = st.empty()
        image_container.image(tif_images, caption=f"Page {frame_num + 1}", use_column_width=True)
        
        # Process the uploaded file using your existing code
        args = parse_args()
        process_button = st.sidebar.button("Process File")
        if process_button:
            result = process_uploaded_file(args, tif_images)

            st.write("Processing completed. Results:")
            table_data = list()
            for key, value in result.items():
                table_data.append([key, value['answer'][0], value['confidence_score'][0]])
            st.table(pd.DataFrame(table_data, columns=['Keywords', 'Values', 'Confidence Score']))


if __name__ == "__main__":
    main()
