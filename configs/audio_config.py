# 请填写/修改对应文件的路径，其中xxx.pth为BERT-VITS2训练得到的模型文件，config.json文件为训练过程中用到的配置文件。具体请参考BERT-VITS2项目的说明。
PROJECTS = {
    "无":{
        "model_path":"",
        "config_path":"",
    },
    "白河萤":{
        "model_path":"./audio/Data/hotaru/models/hotaru_G_900.pth",
        "config_path":"./audio/Data/hotaru/config.json",
    },
    "稻穗信":{
        "model_path":"./audio/Data/shin/models/shin_G_600.pth",
        "config_path":"./audio/Data/shin/config.json",
    },
    "嘉神川克罗艾":{
        "model_path":"./audio/Data/chloe/models/chloe_G_1500.pth",
        "config_path":"./audio/Data/chloe/config.json",
    },
    "嘉神川诺艾儿":{
        "model_path":"./audio/Data/noelle/models/noelle_G_500.pth",
        "config_path":"./audio/Data/noelle/config.json",
    },
    "三城柚莉":{
        "model_path":"./audio/Data/yuzuri/models/yuzuri_G_1000.pth",
        "config_path":"./audio/Data/yuzuri/config.json",
    },
}

DEFAULT_PROJECT = "无"

SDP_RATIO = 0.5
NOISE_SCALE = 0.6
NOISE_SCALE_W = 0.9
LENGTH_SCALE = 1.0

DEFAULT_EMOTION = 'Normal'
DEVICE = 'cuda'