import os
import pprint
import argparse
from PIL import Image, ImageSequence
from predictor import Predictor

# python3 infer-folder.py --model_path_prefix model/fidelity-413/inference


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'dataset')


all_keywords = [
    "Recording Company Name",
    "Recording Doc Number",
    "Recording Document Date",
    "APN",
    "Document Type",
    "Map Book",
    "Map Page",
    "Lot No",
    "Party Name"
]


def generate_images(doc_path):
    images = [os.path.join(doc_path, image) for image in os.listdir(doc_path) if image.endswith('.jpg')]
    return images


def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_path_prefix", type=str, required=True, help="The path prefix of inference model to be used.")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU for inference.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--task_type", default="mrc", type=str, choices=["ner", "cls", "mrc"], help="Specify the task type.")
    parser.add_argument("--lang", default="en", type=str, choices=["ch", "en"], help="Specify the task type.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    args = parser.parse_args()
    # yapf: enable
    return args


def main():
    args = parse_args()
    for dir in os.listdir(INPUT_DIR):
        full_path = os.path.join(INPUT_DIR, dir)
        if os.path.isdir(full_path):
            images = generate_images(full_path)
            
            result = dict.fromkeys(all_keywords, {'answer': [''], 'confidence_score': [0]})
            for image in images:
                if args.task_type == "mrc":
                    args.questions = [all_keywords]
                    docs = [image]
                elif args.task_type == "cls":
                    docs = ["./images/cls_sample.jpg"]
                elif args.task_type == "ner":
                    docs = ["./images/ner_sample.jpg"]
                else:
                    raise ValueError("Unspport task type: {}".format(args.task_type))

                predictor = Predictor(args)
                outputs = predictor.predict(docs)

                result_dict = {}
                for output in outputs[0]['result']:
                    question = output['question']
                    answer = output['answer']
                    confidence_score = output['confidence_score']
                    
                    if question in result_dict.keys():
                        result_dict[question]['answer'].extend(answer)
                        result_dict[question]['confidence_score'].extend(confidence_score)
                    else:
                        result_dict[question] = {'answer': answer, 'confidence_score': confidence_score}

                for key, value in result_dict.items():
                    # breakpoint()
                    if result_dict[key]['confidence_score'] > result[key]['confidence_score']:
                        result[key] = value
    
            pprint.pprint(result)


if __name__ == "__main__":
    main()
