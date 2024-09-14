
import time
import io, os, sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import numpy as np
from flask import Flask, request, Response
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import ffmpeg

from flask_cors import CORS
from flask import make_response

import json

import shutil
import logging
import random
import librosa

cosyvoice = CosyVoice('./pretrained_models/CosyVoice-300M')

spk_new = []

for name in os.listdir("./voices/"):
    print(name.replace(".py",""))
    spk_new.append(name.replace(".py",""))

print("默认音色",cosyvoice.list_avaliable_spks())
print("自定义音色",spk_new)

app = Flask(__name__)

CORS(app, cors_allowed_origins="*")

CORS(app, supports_credentials=True)


def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

@app.route("/generate_audio", methods=['POST'])
def sft_post():
    question_data = request.get_json()

    text = question_data.get('text')
    speaker = question_data.get('speaker')
    customSpeak = question_data.get('customSpeak',0)
    streaming = question_data.get('streaming',0)

    speed = request.args.get('speed',1.0)
    speed = float(speed)
    

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    # 非流式
    if streaming == 0:

        start = time.process_time()
        if not customSpeak:
            output = cosyvoice.inference_sft(text,speaker,"无")
        else:
            output = cosyvoice.inference_sft(text,speaker,speaker)
        end = time.process_time()
        print("infer time:", end - start)
        buffer = io.BytesIO()

        if speed != 1.0:
            try:
                numpy_array = output['tts_speech'].numpy()
                audio = (numpy_array * 32768).astype(np.int16) 
                audio_data = speed_change(audio, speed=speed, sr=int(22050))
                audio_data = torch.from_numpy(audio_data)
                audio_data = audio_data.reshape(1, -1)
            except Exception as e:
                print(f"Failed to change speed of audio: \n{e}")
        else:
            audio_data = output['tts_speech']

        torchaudio.save(buffer,audio_data, 22050, format="wav")
        buffer.seek(0)
        return Response(buffer.read(), mimetype="audio/wav")

    # 流式模式
    else:

        spk_id = speaker

        if customSpeak:
            spk_id = "中文女"

        joblist = cosyvoice.frontend.text_normalize_stream(text,True)

        def generate():
        
            for i in joblist:
                print(i)
                print("流式0")
                tts_speeches = []
                model_input = cosyvoice.frontend.frontend_sft(i, spk_id)
                if customSpeak:
                    # 加载数据
                    newspk = torch.load(f'./voices/{speaker}.pt')

                    model_input["flow_embedding"] = newspk["flow_embedding"]
                    model_input["llm_embedding"] = newspk["llm_embedding"]

                    model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
                    model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

                    model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
                    model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

                    model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
                    model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
                    model_input["prompt_text"] = newspk["prompt_text"]
                    model_input["prompt_text_len"] = newspk["prompt_text_len"]

                model_output = next(cosyvoice.model.inference_stream(**model_input))
                # print(model_input)
                tts_speeches.append(model_output['tts_speech'])
                output = torch.concat(tts_speeches, dim=1)
                buffer = io.BytesIO()
                if speed != 1.0:
                    try:
                        numpy_array = output.numpy()
                        audio = (numpy_array * 32768).astype(np.int16) 
                        audio_data = speed_change(audio, speed=speed, sr=int(22050))
                        audio_data = torch.from_numpy(audio_data)
                        audio_data = audio_data.reshape(1, -1)
                    except Exception as e:
                        print(f"Failed to change speed of audio: \n{e}")
                else:
                    audio_data = output

                torchaudio.save(buffer,audio_data, 22050, format="ogg")
                buffer.seek(0)

                yield buffer.read()

        response = make_response(generate())
        response.headers['Content-Type'] = 'audio/ogg'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
        return response


@app.route("/generate_audio", methods=['GET'])
def sft_get():

    text = request.args.get('text')
    speaker = request.args.get('speaker')
    customSpeak = request.args.get('customerSpeak',0)
    streaming = request.args.get('streaming',0)
    speed = request.args.get('speed',1.0)
    speed = float(speed)

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    # 非流式
    if streaming == 0:

        start = time.process_time()
        if not customSpeak:
            output = cosyvoice.inference_sft(text,speaker,"无")
        else:
            output = cosyvoice.inference_sft(text,speaker,speaker)
        end = time.process_time()
        print("infer time:", end - start)
        buffer = io.BytesIO()

        if speed != 1.0:
            try:
                numpy_array = output['tts_speech'].numpy()
                audio = (numpy_array * 32768).astype(np.int16) 
                audio_data = speed_change(audio, speed=speed, sr=int(22050))
                audio_data = torch.from_numpy(audio_data)
                audio_data = audio_data.reshape(1, -1)
            except Exception as e:
                print(f"Failed to change speed of audio: \n{e}")
        else:
            audio_data = output['tts_speech']

        torchaudio.save(buffer,audio_data, 22050, format="wav")
        buffer.seek(0)
        return Response(buffer.read(), mimetype="audio/wav")

    # 流式模式
    else:

        spk_id = speaker

        if customSpeak:
            spk_id = "中文女"

        joblist = cosyvoice.frontend.text_normalize_stream(text, split=True)

        def generate():
        
            for i in joblist:
                print(i)
                print("流式0")
                tts_speeches = []
                model_input = cosyvoice.frontend.frontend_sft(i, spk_id)
                if customSpeak:
                    # 加载数据
                    newspk = torch.load(f'./voices/{speaker}.pt')

                    model_input["flow_embedding"] = newspk["flow_embedding"]
                    model_input["llm_embedding"] = newspk["llm_embedding"]

                    model_input["llm_prompt_speech_token"] = newspk["llm_prompt_speech_token"]
                    model_input["llm_prompt_speech_token_len"] = newspk["llm_prompt_speech_token_len"]

                    model_input["flow_prompt_speech_token"] = newspk["flow_prompt_speech_token"]
                    model_input["flow_prompt_speech_token_len"] = newspk["flow_prompt_speech_token_len"]

                    model_input["prompt_speech_feat_len"] = newspk["prompt_speech_feat_len"]
                    model_input["prompt_speech_feat"] = newspk["prompt_speech_feat"]
                    model_input["prompt_text"] = newspk["prompt_text"]
                    model_input["prompt_text_len"] = newspk["prompt_text_len"]

                model_output = next(cosyvoice.model.inference_stream(**model_input))
                # print(model_input)
                tts_speeches.append(model_output['tts_speech'])
                output = torch.concat(tts_speeches, dim=1)
                buffer = io.BytesIO()
                if speed != 1.0:
                    try:
                        numpy_array = output.numpy()
                        audio = (numpy_array * 32768).astype(np.int16) 
                        audio_data = speed_change(audio, speed=speed, sr=int(22050))
                        audio_data = torch.from_numpy(audio_data)
                        audio_data = audio_data.reshape(1, -1)
                    except Exception as e:
                        print(f"Failed to change speed of audio: \n{e}")
                else:
                    audio_data = output

                torchaudio.save(buffer,audio_data, 22050, format="ogg")
                buffer.seek(0)

                yield buffer.read()

        response = make_response(generate())
        response.headers['Content-Type'] = 'audio/ogg'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.ogg'
        return response
        
        # return Response(generate(), mimetype='audio/x-wav')


@app.route("/tts_to_audio/", methods=['POST'])
def tts_to_audio():

    import speaker_config
    
    question_data = request.get_json()

    text = question_data.get('text')
    speaker = speaker_config.speaker
    customSpeak = speaker_config.customSpeak

    speed = speaker_config.speed
    

    if not text:
        return {"error": "文本不能为空"}, 400

    if not speaker:
        return {"error": "角色名不能为空"}, 400

    start = time.process_time()
    if not customSpeak:
        output = cosyvoice.inference_sft(text,speaker,"无")
    else:
        output = cosyvoice.inference_sft(text,speaker,speaker)
    end = time.process_time()
    print("infer time:", end - start)
    buffer = io.BytesIO()
    if speed != 1.0:
        try:
            numpy_array = output['tts_speech'].numpy()
            audio = (numpy_array * 32768).astype(np.int16) 
            audio_data = speed_change(audio, speed=speed, sr=int(22050))
            audio_data = torch.from_numpy(audio_data)
            audio_data = audio_data.reshape(1, -1)
        except Exception as e:
            print(f"Failed to change speed of audio: \n{e}")
    else:
        audio_data = output['tts_speech']

    torchaudio.save(buffer,audio_data, 22050, format="wav")
    buffer.seek(0)
    return Response(buffer.read(), mimetype="audio/wav")


@app.route("/custom_spk", methods=['GET'])
def custom_spk():
    spk_new = ["无"]
    for name in os.listdir("./voices/"):
        # print(name.replace(".pt",""))
        spk_new.append(name.replace(".pt",""))
    response = app.response_class(
        response=json.dumps(spk_new),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/speakers", methods=['GET'])
def speakers():

    response = app.response_class(
        response=json.dumps([{"name":"default","vid":1}]),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/default_spk",methods=['GET'])
def default_spk():
    response = app.response_class(
        response=json.dumps(cosyvoice.list_avaliable_spks()),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/clone", methods=['POST'])
def clone():
    question_data = request.get_json()

    tts_text = "我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。"
    prompt_wav = question_data.get('prompt_wav')
    name = question_data.get('name')

    seed = 0
    prompt_text = "创新被定义为带来新想法、新方法、新产品、新服务或新解决方案的过程，从而产生重大的积极影响和价值。"

    logging.info('get zero_shot inference request')
    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
    set_all_random_seed(seed)
    output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)

    shutil.copyfile("./output.pt",f"./voices/{name}.pt")
    response = app.response_class(
        response=json.dumps({"result":"success"}),
        status=200,
        mimetype='application/json'
    )
    return response


max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@app.route("/speakers_list", methods=['GET'])
def speakers_list():

    response = app.response_class(
        response=json.dumps(["female_calm","female","male"]),
        status=200,
        mimetype='application/json'
    )
    return response
    

if __name__ == "__main__":
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)

    app.run(host='0.0.0.0', port=9880)
