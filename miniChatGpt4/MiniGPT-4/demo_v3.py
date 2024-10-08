import argparse
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import cv2
import re
import pickle

# Imports for server
from fastapi import FastAPI, Query, HTTPException , File, UploadFile
from pydantic import BaseModel
from typing import Union
from fastapi.responses import JSONResponse
import logging
from typing import Dict
import uuid



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", type=str, default="eval_configs/minigptv2_eval.yaml", help="path to configuration file.")
    parser.add_argument("--img_path", type=str, default="Kimo.jpg",  help="path to an input image.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def main():

    # ========================================
    #             Model Initialization
    # ========================================

    print('Initializing model')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Model Initialization Finished')

    # upload image
    chat_state = CONV_VISION_minigptv2.copy()
    img_list = []
    

    llm_message = chat.upload_img(args.img_path, chat_state, img_list)
    print(llm_message)


    while True:
        # ask a question
        if len(chat_state.messages) == 1:
            prompt = input("Please enter your prompt \n")
        else:
            prompt = input("Please enter your prompt or type 'exit' to quit \n")
            
            if prompt == "exit":
                break
        

        chat.ask(prompt, chat_state)
        
        if len(chat_state.messages) == 1:
            chat.encode_img(img_list)
        
        # with open('img_list.pkl', 'rb') as f:
        #     img_list = pickle.load(f)
        #     print(img_list)

        
            
        # get answer
        llm_message = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=args.num_beams,
                                temperature=args.temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]

        print("============================================")

        pattern = r"<\d+>"
        
        if re.search(pattern, llm_message):
            image = cv2.imread(args.img_path)

            # Define the coordinates
            coordinates = llm_message.strip("{}<>")

            # Split the coordinates by "><" to get individual values
            x_min, y_min, x_max, y_max = map(int, coordinates.split("><"))

            print("x_min:", x_min)
            print("y_min:", y_min)
            print("x_max:", x_max)
            print("y_max:", y_max)
            
            height, width, _ = image.shape
            x_min_pixel = int(x_min * width / 100)
            y_min_pixel = int(y_min * height / 100)
            x_max_pixel = int(x_max * width / 100)
            y_max_pixel = int(y_max * height / 100)

            # Draw a rectangle around the object
            cv2.rectangle(image, (x_min_pixel, y_min_pixel), (x_max_pixel, y_max_pixel), (255, 0, 0), 2)

            cv2.imwrite('image_with_box.jpg', image)
            print("image_with_box saved")

        print(llm_message)
        # print("============================================")
        # print(chat_state)
        # print("============================================")
        # print(img_list)
        # print("============================================")



if __name__ == "__main__":
    main()