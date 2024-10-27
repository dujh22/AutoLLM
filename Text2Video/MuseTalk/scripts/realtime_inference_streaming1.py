import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
import shutil

import threading
import queue
import time

# 新增导入
from flask import Flask, Response, stream_with_context
import ffmpeg

# 创建Flask应用
app = Flask(__name__)

# 初始化模型
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

# 全局变量，用于帧的传递
frame_queue = queue.Queue()
streaming = False  # 标志位，表示是否正在推理和流式传输

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avatar_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.load_material()
            else:
                print("*********************************")
                print(f"  creating avatar: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.load_material()

    def load_material(self):
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            face_box = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self,
                       res_frame_queue,
                       video_len):
        print(video_len)
        global frame_queue  # 使用全局变量
        while True:
            if self.idx >= video_len - 1:
                frame_queue.put(None)  # 标记结束
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            # 将合成的帧放入全局队列中
            frame_queue.put(combine_frame)
            self.idx = self.idx + 1

    def inference(self,
                  audio_path,
                  fps):
        global streaming
        streaming = True  # 标记开始流式传输
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0
        # 创建一个子线程，处理帧的合成
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num))
        process_thread.start()

        gen = datagen(whisper_chunks,
                      self.input_latent_list_cycle,
                      self.batch_size)
        start_time = time.time()

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                         dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch,
                                      timesteps,
                                      encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        # 等待处理线程完成
        process_thread.join()
        streaming = False  # 标记结束流式传输
        print('Total process time of {} frames = {}s'.format(
            video_num,
            time.time() - start_time))
        print("\n")


def generate_video():
    global frame_queue, streaming
    # 获取帧的宽度和高度（假设所有帧大小一致）
    WIDTH = 256  # 替换为实际宽度
    HEIGHT = 256  # 替换为实际高度

    # 使用FFmpeg进行实时编码
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(WIDTH, HEIGHT), framerate=25)
        .output('pipe:', format='mp4', vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast',
                tune='zerolatency', movflags='frag_keyframe', fflags='+flush_packets', max_delay=0)
        .global_args('-loglevel', 'error')  # 减少日志输出
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, bufsize=10**8)
    )

    # 创建一个独立的线程，从FFmpeg的stdout读取数据并发送给浏览器
    def read_stdout():
        try:
            while True:
                data = process.stdout.read(4096)
                if not data:
                    break
                yield data
        except Exception as e:
            print("Error reading stdout:", e)
        finally:
            process.stdout.close()

    stdout_generator = read_stdout()

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            # 确保帧的尺寸匹配
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            # 将帧转换为RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 写入FFmpeg的stdin
            process.stdin.write(
                frame
                .astype(np.uint8)
                .tobytes()
            )
            # 从FFmpeg的stdout读取数据
            try:
                data = next(stdout_generator)
                if data:
                    yield data
            except StopIteration:
                break
    except Exception as e:
        print("Error in generate_video:", e)
    finally:
        # 关闭FFmpeg进程
        process.stdin.close()
        process.wait()



## 新增函数：生成视频流
#def generate_video():
#    global frame_queue, streaming
#    # 获取帧的宽度和高度（假设所有帧大小一致）
#    WIDTH = 256  # 替换为实际宽度
#    HEIGHT = 256  # 替换为实际高度
#
#    # 使用FFmpeg进行实时编码
#    process = (
#        ffmpeg
#        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(WIDTH, HEIGHT), framerate=25)
#        .output('pipe:', format='mp4', vcodec='libx264', pix_fmt='yuv420p', movflags='frag_keyframe+empty_moov', preset='ultrafast')
#        .run_async(pipe_stdin=True, pipe_stdout=True)
#    )
#
#    while True:
#        frame = frame_queue.get()
#        if frame is None:
#            break
#        # 确保帧的尺寸匹配
#        frame = cv2.resize(frame, (WIDTH, HEIGHT))
#        # 将帧转换为RGB格式
#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        # 写入FFmpeg的stdin
#        process.stdin.write(
#            frame
#            .astype(np.uint8)
#            .tobytes()
#        )
#        # 从FFmpeg的stdout读取数据
#        data = process.stdout.read(1024)
#        if data:
#            yield data
#
#    # 关闭FFmpeg进程
#    process.stdin.close()
#    process.wait()

# 定义Flask路由，提供视频流
@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_video()), mimetype='video/mp4')

if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config",
                        type=str,
                        default="configs/inference/realtime.yaml",
    )
    parser.add_argument("--fps",
                        type=int,
                        default=25,
    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
    )

    args = parser.parse_args()

    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    # 启动Flask服务器
    def start_server():
        app.run(host='0.0.0.0', port=8010, threaded=True)
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        bbox_shift = inference_config[avatar_id]["bbox_shift"]
        audio_dir = inference_config[avatar_id]["audio_dir"]  # 新增，读取audio_dir字段
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=args.batch_size,
            preparation=data_preparation)

        # 动态检测音频目录中的新wav文件
        processed_files = set()
        while True:
            wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
            new_files = [f for f in wav_files if f not in processed_files]
            if new_files:
                for audio_path in new_files:
                    print("Processing new audio file:", audio_path)
                    avatar.inference(audio_path,
                                     args.fps)
                    # 重命名处理过的文件
                    finished_path = audio_path + '.finished'
                    os.rename(audio_path, finished_path)
                    processed_files.add(finished_path)
            else:
                time.sleep(1)  # 等待一段时间后再次检查
    # 等待服务器线程完成
    server_thread.join()

