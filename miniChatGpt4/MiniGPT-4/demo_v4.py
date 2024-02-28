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
import os


app = FastAPI()
sessions = {}

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", type=str, default="eval_configs/minigptv2_eval.yaml", help="path to configuration file.")
    # parser.add_argument("--img_path", type=str, default="Kimo.jpg",  help="path to an input image.")
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

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args() 
    return args

def get_image_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension

# def main():


# ============================================
#             Model Initialization
# ============================================

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

# =========================================================
#             Model Initialization Finished
# =========================================================

class CustomValidationError(Exception):
    def __init__(self, message):
        self.message = message

@app.exception_handler(CustomValidationError)
async def custom_validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "message": exc.message,
            'success': False,
            'data': None
            }
    )

@app.post("/")
def upload_img(image: UploadFile = File(None)):

    if image is None or image.size == 0: 
        raise CustomValidationError("Parameter 'image' is required")

    contents = image.file.read()
    if contents is None: 
        raise CustomValidationError("Parameter 'image' is required")

    global sessions

    session_token = str(uuid.uuid4())

    extenstion = get_image_extension(image.filename)
    generated_filename = f"images/{session_token}{extenstion}"

    with open(f"{generated_filename}", 'wb') as f:
        f.write(contents)
        
    chat_state = CONV_VISION_minigptv2 .copy()
    img_list = []

    chat.upload_img(generated_filename , chat_state, img_list)

    sessions[session_token] = {0 : chat_state , 1 : img_list , 2 : generated_filename}


    return {
            "message": "Session Created Successfully",
            'success': True,
            'data': {
                'session_token': session_token,
            }
        }



@app.get("/")
def getResponse(prompt: str = Query(None) , session_token: str = Query(None)):

    if session_token is None:
        raise CustomValidationError("Parameter 'session_token' is required")

    if prompt is None:
        raise CustomValidationError("Parameter 'prompt' is required")
    elif len(prompt) < 3:
        raise CustomValidationError("Parameter length must be at least 3 characters")


    global sessions
    
    session = sessions.get(session_token , None)
    
    if session is None:
        raise CustomValidationError("Invalid session_token")

    chat_state = session[0]
    img_list = session[1]
    
    chat.ask(prompt, chat_state)
    
    if len(chat_state.messages) == 1:
        chat.encode_img(img_list)


    llm_message = chat.answer(conv=chat_state,
                            img_list=img_list,
                            num_beams=args.num_beams,
                            temperature=args.temperature,
                            max_new_tokens=300,
                            max_length=2000)[0]

    sessions[session_token] = {0 : chat_state , 1 : img_list , 2 : session[2]}
    print(sessions)
    
    pattern = r"<\d+>"
    
    if re.search(pattern, llm_message):

        image_path = session[2]

        image = cv2.imread(image_path)

        # Define the coordinates
        coordinates = llm_message.strip("{}<>")

        # Split the coordinates by "><" to get individual values
        x_min, y_min, x_max, y_max = map(int, coordinates.split("><"))
        
        height, width, _ = image.shape
        x_min_pixel = int(x_min * width / 100)
        y_min_pixel = int(y_min * height / 100)
        x_max_pixel = int(x_max * width / 100)
        y_max_pixel = int(y_max * height / 100)

        # Draw a rectangle around the object
        cv2.rectangle(image, (x_min_pixel, y_min_pixel), (x_max_pixel, y_max_pixel), (255, 0, 0), 2)

        # session_token = str(uuid.uuid4())

        cv2.imwrite('image_with_box.jpg', image)

    print(llm_message)

    return {
            "message": "Response Generated Successfully",
            'success': True,
            'data': llm_message
            }



@app.get("/deleteSession")
def deleteSession(session_token: str = Query(None)):

    
    if session_token is None:
        raise CustomValidationError("Parameter 'session_token' is required")

    global sessions

    session = sessions.get(session_token , None)
    
    if session is None:
        raise CustomValidationError("Invalid session_token")

    image_path = session[2]
    os.remove(image_path)

    sessions.pop(session_token)

    return {
            "message": "Session Deleted Successfully",
            'success': True,
            'data': None
            }

    

# if __name__ == "__main__":
#     main()