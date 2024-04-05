import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
from PIL import Image
import torch
import glob
import pickle
from tqdm import tqdm
import copy
import comfy.utils

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

MuseVCheckPointDir = os.path.join(
    diffusers_path, "TMElyralab/MuseTalk"
)

import sys
sys.path.insert(0,f'{comfy_path}/custom_nodes/ComfyUI-MuseTalk')

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model

from pydub import AudioSegment
import time

# load model weights
audio_processor,vae,unet,pe  = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

class MuseTalkCupAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path":("STRING",{"default":""}),
                "start":("INT",{"default":0,"min":0,"max":3600000}),
                "end":("INT",{"default":1000,"min":0,"max":3600000}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "MuseTalk"

    def run(self,audio_path,start,end):
        sound = AudioSegment.from_file(audio_path)
        sound=sound[start:end]
        t=int(time.time())
        sound.export(f'{comfy_path}/output/{t}.wav', format="wav")
        return (f'{comfy_path}/output/{t}.wav',)

class MuseTalkRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path":("STRING",{"default":""}),
                "audio_path":("STRING",{"default":""}),
                "bbox_shift":("INT",{"default":0}),
                "batch_size":("INT",{"default":8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "MuseTalk"

    def run(self, video_path,audio_path,bbox_shift,batch_size):
        parser = argparse.ArgumentParser()
        parser.add_argument("--bbox_shift",type=int, default=bbox_shift)
        parser.add_argument("--result_dir", default=f'{comfy_path}/output', help="path to output")

        parser.add_argument("--fps",type=int, default=25)
        parser.add_argument("--batch_size",type=int, default=batch_size)
        parser.add_argument("--output_vid_name",type=str,default='')

        args, unknown = parser.parse_known_args()

        task_id="task_0"
        use_saved_coord=False
        fps=args.fps

        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        crop_coord_save_path = os.path.join(args.result_dir, input_basename+".pkl") # only related to video input
        result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
        os.makedirs(result_img_save_path,exist_ok =True)
        
        if args.output_vid_name=="":
            output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
        else:
            output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
        ############################################## extract frames from source video ##############################################
        if get_file_type(video_path)=="video":
            save_dir_full = os.path.join(args.result_dir, input_basename)
            os.makedirs(save_dir_full,exist_ok = True)
            cmd = f"ffmpeg -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        else: # input img folder
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = args.fps
        #print(input_img_list)
        ############################################## extract audio feature ##############################################
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        ############################################## preprocess input image  ##############################################
        if os.path.exists(crop_coord_save_path) and use_saved_coord:
            print("using extracted coordinates")
            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list,args.bbox_shift)
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
                
        i = 0
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)

        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        ############################################## inference batch by batch ##############################################
        print("start inference")
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list = []
        pbar = comfy.utils.ProgressBar(int(np.ceil(float(video_num)/batch_size)))
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            
            tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
            audio_feature_batch = torch.stack(tensor_list).to(unet.device) # torch, B, 5*N,384
            audio_feature_batch = pe(audio_feature_batch)
            
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
            pbar.update(1)
                
        outframes = []
        ############################################## pad to full image ##############################################
        print("pad talking image to original video")
        pbar = comfy.utils.ProgressBar(len(res_frame_list))
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
    #                 print(bbox)
                continue
            
            combine_frame = get_image(ori_frame,res_frame,bbox)
            #cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
            image=Image.fromarray(cv2.cvtColor(combine_frame,cv2.COLOR_BGR2RGB))
            #image=Image.fromarray(np.clip(combine_frame, 0, 255).astype(np.uint8))
            image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0)  # Convert back to CxHxW
            image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
            outframes.append(image_tensor_out)
            pbar.update(1)

        return (torch.cat(tuple(outframes), dim=0),) 

class VHS_FILENAMES_STRING_MuseTalk:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    }
                }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "MuseTalk"
    FUNCTION = "run"

    def run(self, filenames):
        return (filenames[1][-1],)
    
NODE_CLASS_MAPPINGS = {
    "MuseTalkRun":MuseTalkRun,
    "VHS_FILENAMES_STRING_MuseTalk":VHS_FILENAMES_STRING_MuseTalk,
    "MuseTalkCupAudio":MuseTalkCupAudio,
}
