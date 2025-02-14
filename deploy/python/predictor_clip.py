import os
import faiss
import numpy as np
import json
import cv2

from paddleclas.deploy.utils import logger, config
from paddleclas.deploy.utils.predictor import Predictor
from paddleclas.deploy.utils.get_image_list import get_image_list
from paddleclas.deploy.python.preprocess import create_operators
from paddleclas.ppcls.arch.clip.tokenizer import Tokenizer
from paddleclas.ppcls.arch.clip.clip import tokenize

MODEL = ["image-to-text","image-to-image","image_index_build","text_index_build","text-to-image"]

def get_text(path):
    text = None
    if os.path.exists(path):
        with open(path,"r") as f:
            text = f.readlines()
    return text
        
class CLIPPredictor(Predictor):
    def __init__(self, config):
        self.config = config
        self.mode = config["Global"]["mode"]
        assert self.mode in MODEL
        self.args = config["Global"]
        self.embedding_size = config["Global"]["embedding_size"]
        self.inference_image_encoder_dir = config["Global"]["inference_image_encoder_dir"]
        self.inference_text_encoder_dir = config["Global"]["inference_text_encoder_dir"]
        self.clip_tokenizer = Tokenizer()
    
        assert self.args.get("use_onnx", False) == False
        if (self.mode.find("text_index_build") != -1) or (self.mode.find("text-to-image") != -1):
            self.text_encoder, _ = self.create_paddle_predictor(self.args, self.inference_text_encoder_dir)
        else:
            self.image_encoder, _ = self.create_paddle_predictor(self.args, self.inference_image_encoder_dir)

        self.preprocess_ops = create_operators(config["PreProcess"][
            "transform_ops"])

        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.return_k = self.config['IndexProcess']['return_k']

        self.image_index_dir = self.config["IndexProcess"]["image_index_dir"]
        self.text_index_dir = self.config["IndexProcess"]["text_index_dir"]

        if (self.mode == "image-to-image") or (self.mode == "text-to-image"):
            assert self.image_index_dir
        elif self.mode == "image-to-text":
            assert self.text_index_dir

        self.text_searcher = None
        self.image_searcher = None

        if self.mode.find("build") == -1:
            if self.mode == "image-to-image" or self.mode == "text-to-image":
                self.image_searcher, self.image_id_map = self.index_initilizer(self.image_index_dir)
            else:
                self.text_searcher, self.text_id_map = self.index_initilizer(self.text_index_dir)

    
    def index_initilizer(self, path):
        Searcher = None
        assert os.path.exists(os.path.join(
            path, "index.bin")), "vector.index not found ..."
        if self.config['IndexProcess'].get("dist_type") == "hamming":
            Searcher = faiss.read_index_binary(
                os.path.join(path, "index.bin"))
        else:
            Searcher = faiss.read_index(
                os.path.join(path, "index.bin"))
        with open(os.path.join(path,"index.json"),"r") as f:
            id_map = json.load(f)
        return Searcher, id_map
    
    def image_index_builder(self, images, name_list):
        root = os.path.join(self.image_index_dir)
        if os.path.exists(root) == False:
            os.makedirs(root)
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)
        embeddings = self.encoder(image, self.image_encoder)
        index = faiss.IndexFlatL2(self.embedding_size)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(root,"index.bin"))
        with open(os.path.join(root,"index.json", ), "w") as f:
            json.dump(name_list, f)

        
    def text_index_builder(self, text_lists):
        root = os.path.join(self.text_index_dir)
        if os.path.exists(root) == False:
            os.makedirs(root)
        index = faiss.IndexFlatL2(self.embedding_size)
        texts = tokenize(text_lists, self.clip_tokenizer).numpy()
        text_embeddings = self.encoder(texts, self.text_encoder)
        index.add(text_embeddings)
        faiss.write_index(index, os.path.join(root, "index.bin"))
        with open(os.path.join(root, "index.json"), "w") as f:
            json.dump(text_lists, f)
    
    def encoder(self, inputs, predictor):
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        input_tensor.copy_from_cpu(inputs)
        predictor.run()
        embeddings = output_tensor.copy_to_cpu()
        return embeddings

    def predict_image(self, images):
        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)

        embeddings = self.encoder(image, self.image_encoder)

        output = []
        preds = {}
        if self.mode == "image-to-image":
            scores, docs = self.image_searcher.search(embeddings, self.return_k)
            id = [ self.image_id_map[i] for i in docs[0]]
        else:
            scores, docs = self.text_searcher.search(embeddings, self.return_k)
            id = [ self.text_id_map[i] for i in docs[0]]

        for i in range(len(id)):
            score = scores[i][0]
            if score > self.config["IndexProcess"]["score_thres"]:
                preds["score"] = score
                preds["results"] = id[i]
                output.append(preds)
        return output
    
    def predict_texts(self, texts):
        texts = tokenize(texts, self.clip_tokenizer).numpy()
        text_embeddings = self.encoder(texts, self.text_encoder)
        scores, docs = self.image_searcher.search(text_embeddings, self.return_k)
        id = [ self.image_id_map[i] for i in docs[0]]
        output = []
        preds = {}
        for i in range(len(id)):
            score = scores[i][0]
            if score > self.config["IndexProcess"]["score_thres"]:
                preds["score"] = score
                preds["results"] = id[i]
                output.append(preds)
        return output
    
    def predict(self, inputs):
        output = []
        if self.mode == "text-to-image":
            output = self.predict_texts(inputs)
        elif self.model == "image-to-image" or self.mode == "image-to-text":
            output = self.predict_image
        return output

def main(config):
    predictor = CLIPPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])
    text_list = get_text(config["Global"].get("texts",""))
    mode = config["Global"]["mode"]

    batch_imgs = []
    batch_names = []
    for idx, img_path in enumerate(image_list):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
        else:
            img = img[:, :, ::-1]
            batch_imgs.append(img)
            img_name = os.path.basename(img_path)
            batch_names.append(img_name)
    
    if mode == "image-to-image" or mode == "image-to-text":
        output= predictor.predict(batch_imgs)
        print(output)

    elif mode == "image_index_build":
        predictor.image_index_builder(batch_imgs, image_list)
        print("built the image index")

    elif mode == "text-to-image":
        output = predictor.predict(text_list)
        print(output)
    else:
        predictor.text_index_builder(text_list)
        print("built the text index")
    return




        

        