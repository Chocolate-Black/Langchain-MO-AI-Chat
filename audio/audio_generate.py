# flake8: noqa: E402
import gc
import os
import logging
from audio.tools.sentence import split_by_language
import traceback

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
from audio.utils import get_hparams_from_file
from audio.infer import get_net_g, infer_multilang
import gradio as gr
import numpy as np
import librosa
from configs.audio_config import (DEVICE,PROJECTS,DEFAULT_EMOTION)

class AudioGenerator:
    def __init__(self,
                 project:str,
                 config_path:str,
                 device:str = DEVICE) -> None:
        if config_path != "":
            self.hps = get_hparams_from_file(config_path)
        else:
            self.hps = None
        self.device = device
        if device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.project = project
        if project not in PROJECTS.keys():
            raise ValueError(f"{project} does not exist! Please check the config file!")
        self.model_path = PROJECTS[project]['model_path']
        if self.hps != None:
            self.version = self.hps.version
        else:
            self.version = None
        # self.speaker_ids = PROJECTS[project]['spk2id']
        # self.speakers = list(self.speaker_ids.keys())
        self.model = self.load_model()
        
    def change_model(self,project:str):
        if project not in PROJECTS.keys():
            print(f"{project} does not exist! Please check the config file!")
            return
        self.unload_model()
        print('change model path')
        self.model_path = PROJECTS[project]['model_path']
        print('change version')
        self.version = PROJECTS[project]['version']
        self.speaker_ids = PROJECTS[project]['spk2id']
        self.speakers = list(self.speaker_ids.keys())
        print('load model')
        self.model = self.load_model()
        
    
    def load_model(self):
        if self.model_path == "":
            print('load no model!')
            return
        try:
            model = get_net_g(self.model_path,self.version,self.device,self.hps)
            return model
        except Exception as e:
            print(type(e))
            print(str(e))
            traceback.print_exc()
            traceback.print_exc(file=open('log.txt', 'a'))
            return
        
    def unload_model(self):
        del self.model
        self.model = None
        gc.collect()
        free_up_memory()       
        
    def generate_audio_multilang(self,
                       slices,
                       sdp_ratio:float,
                       noise_scale:float,
                       noise_scale_w:float,
                       length_scale:float,
                       speaker,
                       language:str,
                       reference_audio,
                       emotion:str = DEFAULT_EMOTION,
                       skip_start:bool =False,
                       skip_end:bool =False):
        
        audio_list = []
        free_up_memory()

        with torch.no_grad():
            for idx, piece in enumerate(slices):
                skip_start = idx != 0
                skip_end = idx != len(slices) - 1
                audio = infer_multilang(
                    piece,
                    reference_audio=reference_audio,
                    emotion=emotion,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language[idx],
                    hps=self.hps,
                    net_g=self.model,
                    device=self.device,
                    skip_start=skip_start,
                    skip_end=skip_end,
                )
                audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
                audio_list.append(audio16bit)
        return audio_list
    
    def process_text(self,
                     text: str,
                     speaker,
                     sdp_ratio:float,
                     noise_scale:float,
                     noise_scale_w:float,
                     length_scale:float,
                     reference_audio,
                     emotion:str = DEFAULT_EMOTION):
        audio_list = []
        _text, _lang = process_auto(text)
        print(f"Text: {_text}\nLang: {_lang}")
        audio_list.extend(
            self.generate_audio_multilang(
                _text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                _lang,
                reference_audio,
                emotion,
            )
        )
        return audio_list
    
    def convert_text_to_audio(self,
                              text: str,
                              speaker,
                              sdp_ratio:float,
                              noise_scale:float,
                              noise_scale_w:float,
                              length_scale:float,
                              reference_audio= None,
                              emotion:str = DEFAULT_EMOTION,
                              prompt_mode:str = 'Text prompt'):
        
        if self.model == None:
            print('can not convert because no model!')
            return
        
        if prompt_mode == "Audio prompt":
            if reference_audio == None:
                return ("Invalid audio prompt", None)
            else:
                reference_audio = load_audio(reference_audio)[1]
        else:
            reference_audio = None

        audio_list = self.process_text(
            text,
            speaker,
            sdp_ratio,
            noise_scale,
            noise_scale_w,
            length_scale,
            reference_audio,
            emotion
        )

        audio_concat = np.concatenate(audio_list)
        return audio_concat,self.hps.data.sampling_rate

def free_up_memory():
    # Prior inference run might have large variables not cleaned up due to exception during the run.
    # Free up as much memory as possible to allow this run to be successful.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_auto(text):
    _text, _lang = [], []
    for slice in text.split("|"):
        if slice == "":
            continue
        temp_text, temp_lang = [], []
        sentences_list = split_by_language(slice, target_languages=["zh", "ja", "en"])
        for sentence, lang in sentences_list:
            if sentence == "":
                continue
            temp_text.append(sentence)
            if lang == "ja":
                lang = "jp"
            temp_lang.append(lang.upper())
        _text.append(temp_text)
        _lang.append(temp_lang)
    return _text, _lang

def load_audio(path):
    audio, sr = librosa.load(path, 48000)
    return sr, audio

def format_utils(text, speaker):
    _text, _lang = process_auto(text)
    res = f"[{speaker}]"
    for lang_s, content_s in zip(_lang, _text):
        for lang, content in zip(lang_s, content_s):
            res += f"<{lang.lower()}>{content}"
        res += "|"
    return "mix", res[:-1]

def gr_util(item):
    if item == "Text prompt":
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }
    else:
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }

