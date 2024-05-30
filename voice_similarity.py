import argparse
import os
import torchaudio
import torchaudio.transforms as T
import platform
import time
from modelscope.pipelines import pipeline
import gradio as gr
import shutil
import webbrowser
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import subprocess
import sys

MODELS = {
    'speech_campplus_sv_zh-cn_16k-common': {
        'task': 'speaker-verification',
        'model': 'models/speech_campplus_sv_zh-cn_16k-common',
        'model_revision': 'v1.0.0'
    }
}


def batch_clean_paths(paths):
    """
    批量处理路径列表，对每个路径调用 clean_path() 函数。

    参数:
        paths (list[str]): 包含待处理路径的列表。

    返回:
        list[str]: 经过 clean_path() 处理后的路径列表。
    """
    cleaned_paths = []
    for path in paths:
        cleaned_paths.append(clean_path(path))
    return cleaned_paths


def clean_path(path_str):
    if platform.system() == 'Windows':
        path_str = path_str.replace('/', '\\')
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ").strip("\u202a")


class Similarity:
    def __init__(self, score, path):
        self.score = score
        self.path = path

    def __repr__(self):
        return f"Similarity(score={self.score}, path='{self.path}')"


class SimilarityManager:
    filename = 'similarity.txt'  # 静态变量，保存相似度的文件名

    @staticmethod
    def save_to_file(similarities: list[Similarity]):
        # 确保输入的similarities是Similarity对象的列表
        if not all(isinstance(item, Similarity) for item in similarities):
            raise ValueError("All items in the similarities list must be instances of Similarity")

        # Step 5: 将排序后的结果写入输出结果文件（支持中文）
        formatted_scores = [f'{item.score}|{clean_path(item.path)}' for item in similarities]
        with open(SimilarityManager.filename, 'w', encoding='utf-8') as f:
            # 使用'\n'将每个字符串分开，使其写入不同行
            content = '\n'.join(formatted_scores)
            f.write(content)

    @staticmethod
    def load_from_file() -> list[Similarity]:
        audio_list: list[Similarity] = []

        try:
            with open(SimilarityManager.filename, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"File {SimilarityManager.filename} not found.")
            return audio_list

        # 遍历每一行，检查分值并移动文件
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) != 2:
                print(f"警告: 无效的行格式 - {line}")
                continue

            try:
                audio_score = float(parts[0])
            except ValueError:
                print(f"警告: 无效的分数 - {parts[0]}")
                continue

            audio_path = clean_path(parts[1])
            audio_list.append(Similarity(audio_score, audio_path))

        return audio_list


def init_model(model_type='speech_campplus_sv_zh-cn_16k-common'):
    return pipeline(
        task=MODELS[model_type]['task'],
        model=MODELS[model_type]['model'],
        model_revision=MODELS[model_type]['model_revision']
    )


SV_PIPELINE = init_model()


def compare_audio_and_generate_report(reference_audio_path, comparison_dir_path):
    infos = []

    # 记录开始时间
    start_time = time.time()

    last_part = os.path.basename(comparison_dir_path)

    sampling_dir = os.path.join('重采样', last_part)

    if not os.path.exists(sampling_dir):
        os.makedirs(sampling_dir)

    # Step 1: 获取比较音频目录下所有音频文件的路径
    comparison_audio_paths = [os.path.join(comparison_dir_path, f) for f in os.listdir(comparison_dir_path) if
                              f.endswith('.wav')]

    for audio_path in comparison_audio_paths:
        sampling_audio = audio_path.replace(comparison_dir_path, sampling_dir)
        if not os.path.exists(sampling_audio):
            # 如果没有重采样后的音频，则进行重采样
            ensure_16k_wav(audio_path, 16000, sampling_dir)

    # 记录结束时间
    end_time = time.time()

    # 计算总时间并转换为秒
    execution_time = end_time - start_time

    if platform.system() == 'Windows':
        # 因为这个模型是基于16k音频数据训练的，为了避免后续比较时，每次都对参考音频进行重采样，所以，提前进行了采样
        # windows不支持torchaudio.sox_effects.apply_effects_tensor，所以改写了依赖文件中的重采样方法
        # 改用torchaudio.transforms.Resample进行重采样，如果在非windows环境下，没有更改依赖包的采样方法的话，
        # 使用这段代码进行预采样会出现因为采样方法不同，而导致的模型相似度计算不准确的问题
        # 当然如果在windows下，使用了其他的采样方法，也会出现不准确的问题
        reference_audio_16k = ensure_16k_wav(reference_audio_path)
    else:
        reference_audio_16k = reference_audio_path

    # 记录开始时间
    compare_start_time = time.time()

    # Step 2: 用参考音频依次比较音频目录下的每个音频，获取相似度分数及对应路径
    all_count = len(comparison_audio_paths)
    has_processed_count = 0
    similarity_scores: list[Similarity] = []
    for audio_path in comparison_audio_paths:
        try:
            sampling_audio = audio_path.replace(comparison_dir_path, sampling_dir)
            score = SV_PIPELINE([reference_audio_16k, sampling_audio])['score']
            similarity_scores.append(Similarity(score, audio_path))
            has_processed_count += 1
            print(f'进度：{has_processed_count}/{all_count}')
        except Exception as e:
            print(f'比较音频 {audio_path} 时出错：{e}')

    # 记录开始时间
    compare_end_time = time.time()

    # 计算总时间并转换为秒
    compare_execution_time = compare_end_time - compare_start_time

    # Step 3: 根据相似度分数降序排列
    similarity_scores.sort(key=lambda x: x.score, reverse=True)

    SimilarityManager.save_to_file(similarity_scores)

    # 将信息格式化为一个字符串
    execution_info = (
        f"重采样执行时间: {execution_time:.2f} 秒\n"
        f"音频比较执行时间: {compare_execution_time:.2f} 秒"
    )

    # 添加到 infos 列表
    infos.append(execution_info)

    # 打印信息
    print(execution_info)

    return '\n'.join(infos)


def ensure_16k_wav(audio_file_path, target_sample_rate=16000, temp_dir='temp'):
    """
    输入一个音频文件地址，判断其采样率并决定是否进行重采样，然后将结果保存到指定的输出文件。

    参数:
        audio_file_path (str): 音频文件路径。
        output_file_path (str): 保存重采样后音频数据的目标文件路径。
        target_sample_rate (int, optional): 目标采样率，默认为16000Hz。
    """
    # 读取音频文件并获取其采样率
    waveform, sample_rate = torchaudio.load(audio_file_path)

    # 判断是否需要重采样
    if sample_rate == target_sample_rate:
        return audio_file_path
    else:

        # 创建Resample实例
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)

        # 应用重采样
        resampled_waveform = resampler(waveform)

        # 创建临时文件夹
        os.makedirs(temp_dir, exist_ok=True)

        # 设置临时文件名
        temp_file_path = os.path.join(temp_dir, os.path.basename(audio_file_path))

        # 保存重采样后的音频到指定文件
        torchaudio.save(temp_file_path, resampled_waveform, target_sample_rate)

    return temp_file_path


def move_high_score_audios(score_threshold, target_dir):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    audio_list: list[Similarity] = SimilarityManager.load_from_file()

    # 遍历每一行，检查分值并移动文件
    for audio in audio_list:

        audio_score = audio.score
        audio_path = audio.path

        # 检查分值是否高于阈值
        if audio_score >= score_threshold:
            # 构建目标路径
            target_audio_path = os.path.join(target_dir, os.path.basename(audio_path))

            # 移动文件
            try:
                shutil.move(audio_path, target_audio_path)
                print(f"已移动: {audio_path} -> {target_audio_path}")
            except Exception as e:
                print(f"错误移动文件: {audio_path} - 原因: {e}")


def compare_audio_and_generate_report_event(reference_audio_path, comparison_dir_path):
    reference_audio_path, comparison_dir_path \
        = batch_clean_paths([reference_audio_path, comparison_dir_path])

    text_compair_info = None

    try:
        if reference_audio_path is None or reference_audio_path == '':
            raise Exception("请指定一个有效的参考音频文件路径")
        if comparison_dir_path is None or comparison_dir_path == '':
            raise Exception("请指定一个有效的比较音频目录路径")

        text_compair_info = compare_audio_and_generate_report(reference_audio_path, comparison_dir_path)

    except Exception as e:
        print(f"错误: {e}")
        text_compair_info = f"发生异常：{e}"

    return text_compair_info


def move_high_score_audios_event(text_split_score, text_move_dir):
    text_move_dir = clean_path(text_move_dir)

    text_move_info = None
    try:

        text_split_score = float(text_split_score)

        if text_split_score is None or text_split_score == '':
            raise Exception("请指定一个有效的分值")

        if text_move_dir is None or text_move_dir == '':
            raise Exception("请指定一个有效的目标目录")

        move_high_score_audios(text_split_score, text_move_dir)

    except Exception as e:
        print(f"错误: {e}")
        text_move_info = f"发生异常：{e}"

    return text_move_info


def open_html_event():
    # 定位到HTML文件的路径
    file_path = 'show_audio.html'

    # 如果HTML文件和这个Python脚本不在同一个目录，你需要指定完整的路径
    # 例如：file_path = '/path/to/your/html/file/example.html'

    # 将文件路径转换为适用于URL的路径形式
    html_file_url = 'file://' + os.path.realpath(file_path)

    # 使用默认浏览器打开HTML文件
    webbrowser.open(html_file_url)


def open_file(file_path):
    file_path = clean_path(file_path)
    if sys.platform.startswith('darwin'):
        subprocess.run(['open', file_path])  # macOS
    elif os.name == 'nt':  # For Windows
        os.startfile(file_path)
    elif os.name == 'posix':  # For Linux, Unix, etc.
        subprocess.run(['xdg-open', file_path])


def create_gradio_interface():
    with gr.Blocks() as gradio_app:
        gr.Markdown(value="基本介绍：这是一个声纹识别工具，也可以做情绪分离。")
        with gr.Row():
            text_refer_audio = gr.Textbox(label="输入参考音频文件路径", placeholder="请输入参考音频文件路径", scale=4)
            text_compair_dir = gr.Textbox(label="输入比较音频文件夹路径", placeholder="请输入比较音频文件夹路径",
                                          scale=4)
            button_compair_dir_open = gr.Button("打开文件", variant="primary", scale=1)
            button_compair_dir_open.click(open_file, [text_compair_dir], [])
        with gr.Row():
            button_compair = gr.Button(value="开始比较")
            text_compair_info = gr.Text(label="比较结果", value="", interactive=False)
            button_compair.click(
                compare_audio_and_generate_report_event,
                inputs=[text_refer_audio, text_compair_dir],
                outputs=[text_compair_info],
            )
        with gr.Row():
            button_open_report = gr.Button(value="打开报告结果")
            button_open_report.click(
                open_html_event,
                inputs=[],
                outputs=[],
            )
        with gr.Row():
            text_split_score = gr.Textbox(label="请输入待分割分值", placeholder="请输入分值", scale=4)
            text_move_dir = gr.Textbox(label="请输入待移动文件夹路径", placeholder="请输入待移动文件夹路径", scale=4)
            button_move_dir_open = gr.Button("打开文件", variant="primary", scale=1)
            button_move_dir_open.click(open_file, [text_move_dir], [])
        with gr.Row():
            button_start_move = gr.Button(value="开始移动")
            text_move_info = gr.Text(label="移动结果", value="", interactive=False)
            button_start_move.click(
                move_high_score_audios_event,
                inputs=[text_split_score, text_move_dir],
                outputs=[text_move_info],
            )
    return gradio_app


# 使用 Gradio 生成 FastAPI 应用
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

app = gr.mount_gradio_app(app, create_gradio_interface(), path="/gradio")


# FastAPI额外的路由可以在这里定义
@app.get("/getFile")
def read_root():
    return SimilarityManager.load_from_file()


if __name__ == '__main__':
    url = "http://127.0.0.1:8031/gradio"
    webbrowser.open(url)

    uvicorn.run(app, host='127.0.0.1', port=8031, workers=1)
