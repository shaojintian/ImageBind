#!/usr/bin/env python3
# 这是一个 "shebang" 行，通常用于 Unix-like 系统，
# 指示系统使用 python3 解释器来执行这个脚本。

# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 版权声明，指出部分代码由 Meta Platforms, Inc. 及其附属公司拥有版权。

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 许可证声明，说明代码的开源许可证信息通常在项目根目录的 LICENSE 文件中。

import logging # 导入日志模块，用于记录程序运行时的信息。
import math    # 导入数学模块，提供数学运算功能，如 ceil (向上取整)。
import pkg_resources # 导入 pkg_resources 模块，Setuptools 的一部分，用于在运行时发现和使用包资源。

import torch             # 导入 PyTorch 核心库。
import torch.nn as nn    # 导入 PyTorch 的神经网络模块，通常简写为 nn。
import torchaudio        # 导入 PyTorch 的音频处理库。
from PIL import Image    # 从 Pillow (PIL Fork) 库导入 Image 模块，用于图像处理。
from pytorchvideo import transforms as pv_transforms # 从 PyTorchVideo 库导入 transforms 模块，并重命名为 pv_transforms，用于视频数据增强。
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler # 从 PyTorchVideo 导入用于视频剪辑采样的类。
from pytorchvideo.data.encoded_video import EncodedVideo # 从 PyTorchVideo 导入用于处理已编码视频的类。
from torchvision import transforms # 从 TorchVision 库导入 transforms 模块，用于图像数据增强。
from torchvision.transforms._transforms_video import NormalizeVideo # 从 TorchVision 导入用于视频归一化的类。

from imagebind.models.multimodal_preprocessors import SimpleTokenizer # 从项目内部的 imagebind.models.multimodal_preprocessors 模块导入 SimpleTokenizer 类，用于文本分词。

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds
# 定义一个常量，表示音频处理中帧移的默认值，单位是毫秒。

def return_bpe_path():
    # 定义一个函数，用于返回 BPE (Byte Pair Encoding) 词汇表文件的路径。
    return pkg_resources.resource_filename(
        "imagebind", "bpe/bpe_simple_vocab_16e6.txt.gz"
    )
    # 使用 pkg_resources.resource_filename 获取 "imagebind" 包内 "bpe/bpe_simple_vocab_16e6.txt.gz" 文件的绝对路径。
    # 这样做的好处是，无论包安装在哪里，都能正确找到这个资源文件。

def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # 定义一个函数，将原始音频波形 (waveform) 转换为梅尔频谱图 (melspectrogram)。
    # 参数:
    #   waveform: 音频波形张量。
    #   sample_rate: 音频的采样率。
    #   num_mel_bins: 梅尔频谱图的梅尔带数量。
    #   target_length: 目标输出梅尔频谱图的时间帧长度。

    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    # 注释，说明此函数的实现参考了某个 GitHub 项目的代码。

    waveform -= waveform.mean() # 对波形进行去均值操作，中心化数据。
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,            # 输入波形。
        htk_compat=True,     # 启用 HTK (Hidden Markov Model Toolkit) 兼容模式。
        sample_frequency=sample_rate, # 波形的采样频率。
        use_energy=False,    # 不使用能量项作为第一个系数。
        window_type="hanning",# 使用 Hanning 窗函数。
        num_mel_bins=num_mel_bins, # 梅尔带的数量。
        dither=0.0,          # 抖动系数，0.0 表示不使用抖动。
        frame_length=25,     # 帧长度，单位毫秒。
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS, # 帧移，单位毫秒。
    )
    # 使用 torchaudio.compliance.kaldi.fbank 计算梅尔滤波器组能量（fbank），这是梅尔频谱图的一种形式。

    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1) # 将 fbank 的维度从 [num_frames, mel_bins] 转置为 [mel_bins, num_frames]。

    # Pad to target_length
    n_frames = fbank.size(1) # 获取当前梅尔频谱图的时间帧数。
    p = target_length - n_frames # 计算需要填充或截断的帧数。

    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2: # 如果填充/截断的比例超过 20%。
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
        # 记录一条警告日志，提示音频帧数和目标长度之间差异过大。

    # cut and pad
    if p > 0: # 如果需要填充 (p 为正)。
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
        # 使用 torch.nn.functional.pad 在时间维度 (最后一个维度) 的右侧填充 p 帧，填充值为 0。
    elif p < 0: # 如果需要截断 (p 为负)。
        fbank = fbank[:, 0:target_length]
        # 在时间维度上截取前 target_length 帧。

    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0) # 在最前面增加一个维度，形状变为 [1, mel_bins, num_frames]，
                               # 使其类似于单通道图像的格式，方便后续卷积等操作。
    return fbank # 返回处理后的梅尔频谱图。

def get_clip_timepoints(clip_sampler, duration):
    # 定义一个函数，根据给定的剪辑采样器 (clip_sampler) 和视频总时长 (duration)，获取所有剪辑的时间点。
    # 参数:
    #   clip_sampler: 一个剪辑采样器对象，如 ConstantClipsPerVideoSampler。
    #   duration: 视频的总时长（秒）。

    # Read out all clips in this video
    all_clips_timepoints = [] # 初始化一个列表，用于存储所有剪辑的 (开始时间, 结束时间) 元组。
    is_last_clip = False      # 初始化一个标志，表示是否已到达最后一个剪辑。
    end = 0.0                 # 初始化当前剪辑的结束时间，从视频开头 (0.0秒) 开始。
    while not is_last_clip: # 循环直到采样到最后一个剪辑。
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        # 调用 clip_sampler 对象。它会根据上一个剪辑的结束时间 `end` 和视频总时长 `duration` 来确定下一个剪辑的开始和结束时间。
        # `annotation=None` 表示这里不使用额外的标注信息来指导采样。
        # clip_sampler 返回: (clip_start_sec, clip_end_sec, clip_index, aug_index, is_last_clip)
        # 我们这里只关心 start, end, 和 is_last_clip。
        all_clips_timepoints.append((start, end)) # 将获取到的剪辑 (开始时间, 结束时间) 添加到列表中。
    return all_clips_timepoints # 返回包含所有剪辑时间点的列表。

def load_and_transform_vision_data(image_paths, device):
    # 定义一个函数，加载图像数据并进行预处理。
    # 参数:
    #   image_paths: 包含图像文件路径的列表。
    #   device: 指定数据加载到的设备 (例如 'cuda' 或 'cpu')。

    if image_paths is None: # 如果图像路径列表为空。
        return None         # 直接返回 None。

    image_outputs = [] # 初始化一个列表，用于存储处理后的图像张量。

    data_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            # 将图像短边缩放到 224 像素，使用双三次插值。
            transforms.CenterCrop(224),
            # 从图像中心裁剪出 224x224 大小的区域。
            transforms.ToTensor(),
            # 将 PIL Image 对象或 numpy.ndarray 转换为 PyTorch 张量，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]。
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
            # 对图像张量进行归一化，使用指定的均值和标准差。这些值通常是 ImageNet 数据集的统计值。
        ]
    )
    # 定义一系列图像预处理操作。

    for image_path in image_paths: # 遍历输入的每个图像路径。
        with open(image_path, "rb") as fopen: # 以二进制只读模式 ("rb") 打开图像文件。
            image = Image.open(fopen).convert("RGB")
            # 使用 PIL.Image.open 打开图像，并使用 .convert("RGB") 确保图像是 RGB 格式（即使原始图像是灰度或有 alpha 通道）。

        image = data_transform(image).to(device) # 应用定义好的 data_transform 对图像进行预处理，并将结果张量移动到指定的 device。
        image_outputs.append(image) # 将处理后的图像张量添加到列表中。
    return torch.stack(image_outputs, dim=0) # 将列表中的所有图像张量堆叠成一个新的张量，
                                             # dim=0 表示在新的第0维度上堆叠（即批次维度）。返回这个批次图像张量。

def load_and_transform_text(text, device):
    # 定义一个函数，加载文本数据并进行预处理（分词和转换为 token ID）。
    # 参数:
    #   text: 包含文本字符串的列表。
    #   device: 指定数据加载到的设备。

    if text is None: # 如果文本列表为空。
        return None   # 直接返回 None。

    tokenizer = SimpleTokenizer(bpe_path=return_bpe_path())
    # 初始化 SimpleTokenizer，并传入 BPE 词汇表文件的路径。
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    # 对列表中的每个文本字符串 `t`：
    # 1. `tokenizer(t)`: 使用分词器将其转换为 token ID 序列的张量。
    # 2. `.unsqueeze(0)`: 在第0维增加一个维度，使其形状变为 [1, sequence_length]，方便后续批处理。
    # 3. `.to(device)`: 将 token 张量移动到指定的 device。
    # 结果是一个包含多个 [1, sequence_length] 张量的列表。
    tokens = torch.cat(tokens, dim=0) # 将列表中的所有 token 张量在第0维度（批次维度）上拼接起来，
                                     # 形成一个 [batch_size, sequence_length] 的张量。
    return tokens # 返回批次的 token ID 张量。

def load_and_transform_audio_data(
    audio_paths,            # 包含音频文件路径的列表。
    device,                 # 指定数据加载到的设备。
    num_mel_bins=128,       # 梅尔频谱图的梅尔带数量，默认 128。
    target_length=204,      # 梅尔频谱图的目标时间帧长度，默认 204。
    sample_rate=16000,      # 目标采样率，默认 16000 Hz。
    clip_duration=2,        # 每个音频剪辑的持续时间（秒），默认 2 秒。
    clips_per_video=3,      # 每个音频文件采样的剪辑数量，默认 3 个。 (注意: 参数名是 clips_per_video，但这里用于音频)
    mean=-4.268,            # 梅尔频谱图归一化的均值，默认 -4.268。
    std=9.138,              # 梅尔频谱图归一化的标准差，默认 9.138。
):
    # 定义一个函数，加载音频数据并进行预处理（转换为梅尔频谱图并归一化）。
    if audio_paths is None: # 如果音频路径列表为空。
        return None         # 直接返回 None。

    audio_outputs = [] # 初始化一个列表，用于存储处理后的音频梅尔频谱图。
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    # 初始化 ConstantClipsPerVideoSampler，用于从每个音频文件中采样固定数量的、固定时长的剪辑。

    for audio_path in audio_paths: # 遍历输入的每个音频路径。
        waveform, sr = torchaudio.load(audio_path) # 使用 torchaudio.load 加载音频文件，返回波形 (waveform) 和原始采样率 (sr)。
        if sample_rate != sr: # 如果原始采样率与目标采样率不同。
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
            # 使用 torchaudio.functional.resample 将波形重采样到目标采样率。
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        # 调用之前定义的 get_clip_timepoints 函数，获取当前音频中所有剪辑的时间点。
        # 注意：waveform.size(1) 是波形的长度（样本数），除以 sample_rate 得到音频总时长（秒）。

        all_clips = [] # 初始化一个列表，用于存储当前音频文件的所有处理后的剪辑（梅尔频谱图）。
        for clip_timepoints in all_clips_timepoints: # 遍历每个剪辑的时间点。
            waveform_clip = waveform[
                :, # 取所有通道 (通常音频是单声道或双声道)
                int(clip_timepoints[0] * sample_rate) : int( # 计算剪辑开始的样本索引
                    clip_timepoints[1] * sample_rate        # 计算剪辑结束的样本索引
                ),
            ]
            # 从完整波形中提取当前剪辑对应的波形片段。

            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            # 调用之前定义的 waveform2melspec 函数，将剪辑波形转换为梅尔频谱图。
            all_clips.append(waveform_melspec) # 将处理后的梅尔频谱图添加到列表中。

        normalize = transforms.Normalize(mean=mean, std=std)
        # 定义一个归一化操作，使用给定的均值和标准差。
        all_clips = [normalize(ac).to(device) for ac in all_clips]
        # 对列表中的每个梅尔频谱图剪辑 `ac`：
        # 1. `normalize(ac)`: 进行归一化。
        # 2. `.to(device)`: 将结果张量移动到指定的 device。

        all_clips = torch.stack(all_clips, dim=0) # 将当前音频文件的所有梅尔频谱图剪辑堆叠成一个新的张量（在第0维，即剪辑数量维度）。
        audio_outputs.append(all_clips) # 将这个包含多个剪辑的张量添加到 audio_outputs 列表中。

    return torch.stack(audio_outputs, dim=0) # 将列表中所有音频文件的处理结果（每个结果本身是一个包含多个剪辑的张量）
                                             # 再次堆叠，形成一个 [batch_size, num_clips_per_audio, 1, num_mel_bins, target_length] 的张量。

def crop_boxes(boxes, x_offset, y_offset):
    # 定义一个函数，根据给定的 x 和 y 偏移量裁剪边界框 (bounding boxes)。
    # 参数:
    #   boxes (ndarray or None): 边界框，形状为 [num_boxes, 4]。每行格式通常是 [x_min, y_min, x_max, y_max]。
    #   x_offset (int): x 轴上的裁剪偏移量。
    #   y_offset (int): y 轴上的裁剪偏移量。
    # 返回:
    #   cropped_boxes (ndarray or None): 裁剪后的边界框。

    cropped_boxes = boxes.copy() # 复制原始边界框，以避免修改原始数据。
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset # 边界框的 x 坐标 (x_min, x_max) 减去 x_offset。
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset # 边界框的 y 坐标 (y_min, y_max) 减去 y_offset。

    return cropped_boxes # 返回裁剪后的边界框。

def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    # 定义一个函数，对图像（或视频帧）进行均匀的空间裁剪，并可选地裁剪相应的边界框。
    # 参数:
    #   images (tensor): 输入图像，形状通常为 [num_frames, channel, height, width] 或 [channel, height, width]。
    #   size (int): 裁剪后的正方形目标尺寸 (height 和 width 都为 size)。
    #   spatial_idx (int): 空间裁剪索引。0, 1, 或 2，分别对应左/上、中、右/下裁剪。
    #   boxes (ndarray or None): 可选，与图像对应的边界框。
    #   scale_size (int): 可选，如果提供，则在裁剪前将图像短边缩放到此尺寸。
    # 返回:
    #   cropped (tensor): 裁剪后的图像。
    #   cropped_boxes (ndarray or None): 裁剪后的边界框。

    assert spatial_idx in [0, 1, 2] # 断言 spatial_idx 必须是 0, 1, 或 2。
    ndim = len(images.shape) # 获取输入图像张量的维度数。
    if ndim == 3: # 如果是单张图像 (channel, height, width)。
        images = images.unsqueeze(0) # 在前面增加一个批次/帧维度，变为 (1, channel, height, width)。
    height = images.shape[2] # 获取图像的高度。
    width = images.shape[3]  # 获取图像的宽度。

    if scale_size is not None: # 如果指定了 scale_size。
        if width <= height: # 如果宽度小于等于高度 (即宽度是短边或图像是方形/竖直)。
            width, height = scale_size, int(height / width * scale_size) # 将宽度缩放到 scale_size，高度按比例缩放。
        else: # 如果高度小于宽度 (即高度是短边或图像是水平)。
            width, height = int(width / height * scale_size), scale_size # 将高度缩放到 scale_size，宽度按比例缩放。
        images = torch.nn.functional.interpolate(
            images,                  # 输入图像张量。
            size=(height, width),    # 目标尺寸。
            mode="bilinear",         # 使用双线性插值。
            align_corners=False,     # align_corners 参数，通常设为 False。
        )
        # 使用双线性插值将图像缩放到新的 (height, width)。

    y_offset = int(math.ceil((height - size) / 2)) # 计算中心裁剪时 y 轴的起始偏移量。
    x_offset = int(math.ceil((width - size) / 2))  # 计算中心裁剪时 x 轴的起始偏移量。

    if height > width: # 如果图像是竖直方向较长。
        if spatial_idx == 0: # 对应顶部裁剪。
            y_offset = 0
        elif spatial_idx == 2: # 对应底部裁剪。
            y_offset = height - size
    else: # 如果图像是水平方向较长或方形。
        if spatial_idx == 0: # 对应左侧裁剪。
            x_offset = 0
        elif spatial_idx == 2: # 对应右侧裁剪。
            x_offset = width - size
    # 根据 spatial_idx 调整 x_offset 和 y_offset，以实现左/上、中、右/下裁剪。
    # 如果 spatial_idx 是 1 (中间)，则使用上面计算的中心裁剪偏移量。

    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    # 对图像进行切片操作，提取裁剪区域。
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    # 如果提供了边界框，则调用 crop_boxes 函数对边界框进行相应的裁剪。
    if ndim == 3: # 如果原始输入是单张图像。
        cropped = cropped.squeeze(0) #移除之前添加的批次/帧维度。
    return cropped, cropped_boxes # 返回裁剪后的图像和边界框。

class SpatialCrop(nn.Module):
    # 定义一个 PyTorch 模块，用于对视频进行空间裁剪。
    # 这个模块通常在时间维度裁剪之后使用，将每个时间剪辑在空间上裁剪成多个视图。
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """
    # 文档字符串解释了该类的用途和使用场景。

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        # 构造函数。
        # 参数:
        #   crop_size (int): 空间裁剪的目标尺寸，默认 224。
        #   num_crops (int): 要生成的空间裁剪数量，默认 3 (左/上、中、右/下)。
        super().__init__() # 调用父类 nn.Module 的构造函数。
        self.crop_size = crop_size # 保存裁剪尺寸。
        if num_crops == 3: # 如果需要 3 个裁剪。
            self.crops_to_ext = [0, 1, 2] # 定义要提取的裁剪索引 (对应 spatial_idx)。
            self.flipped_crops_to_ext = [] # 定义对水平翻转图像提取的裁剪索引 (这里为空，表示不进行翻转裁剪)。
        elif num_crops == 1: # 如果只需要 1 个裁剪 (通常是中心裁剪)。
            self.crops_to_ext = [1] # 只提取中心裁剪。
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet") # 其他裁剪数量暂不支持。

    def forward(self, videos):
        # 定义前向传播函数。
        # 参数:
        #   videos: 一个列表，其中每个元素是一个视频剪辑张量，形状为 (C, T, H, W) - 通道, 帧数, 高度, 宽度。
        # 返回:
        #   videos: 一个列表，其元素数量是输入列表的 num_crops 倍。每个视频被空间裁剪成 (C, T, H', W')。

        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        # 断言输入必须是一个列表。
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        # 断言列表中的每个视频张量都必须是4维的 (C,T,H,W)。
        res = [] # 初始化一个空列表，用于存储所有空间裁剪后的视频剪辑。
        for video in videos: # 遍历输入列表中的每个视频剪辑。
            for spatial_idx in self.crops_to_ext: # 遍历定义的裁剪索引。
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
                # 调用 uniform_crop 函数进行空间裁剪，只取返回的图像部分 ([0])，并添加到结果列表中。
            if not self.flipped_crops_to_ext: # 如果不需要进行翻转裁剪。
                continue # 继续处理下一个视频。
            # (以下代码块在此配置下不会执行，因为 self.flipped_crops_to_ext 为空)
            flipped_video = transforms.functional.hflip(video) # 对视频进行水平翻转。
            for spatial_idx in self.flipped_crops_to_ext: # 遍历翻转裁剪的索引。
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
                # 对翻转后的视频进行空间裁剪。
        return res # 返回包含所有空间裁剪结果的列表。

def load_and_transform_video_data(
    video_paths,       # 包含视频文件路径的列表。
    device,            # 指定数据加载到的设备。
    clip_duration=2,   # 每个视频剪辑的持续时间（秒），默认 2 秒。
    clips_per_video=5, # 每个视频文件采样的剪辑数量，默认 5 个。
    sample_rate=16000, # 视频解码时音频的采样率 (虽然这里 decode_audio=False，但参数仍存在)。
):
    # 定义一个函数，加载视频数据并进行预处理。
    if video_paths is None: # 如果视频路径列表为空。
        return None         # 直接返回 None。

    video_outputs = [] # 初始化一个列表，用于存储处理后的视频数据。
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224), # PyTorchVideo 的变换：将视频帧的短边缩放到 224 像素。
            NormalizeVideo( # TorchVision 的变换：对视频帧进行归一化。
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    # 定义视频帧的预处理操作序列。

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    # 初始化 ConstantClipsPerVideoSampler，用于从每个视频中采样固定数量的、固定时长的剪辑（时间维度上的采样）。
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)
    # 初始化 PyTorchVideo 的 UniformTemporalSubsample，用于从每个时间剪辑中均匀采样指定数量的帧。
    # 注意:这里的 `num_samples=clip_duration` 可能是一个笔误或特定用法。
    # 通常 `num_samples` 会是一个整数，比如 8 或 16 帧。如果 `clip_duration` 是秒数 (如 2)，
    # 那么这里可能意味着采样 2 帧，或者这个参数的含义与名称不完全对应，需要结合上下文理解。
    # 假设这里 `clip_duration` 应该是一个代表帧数的整数，或者 `UniformTemporalSubsample` 对此有特殊处理。
    # **更新理解**: 从上下文看，`clip_duration` 在 `ConstantClipsPerVideoSampler` 中是秒，
    # 而在 `UniformTemporalSubsample` 中作为 `num_samples`，这通常意味着从该时间段的剪辑中采样 `clip_duration` 数量的帧。
    # 如果 `clip_duration` 是 2 (秒)，而视频帧率是 (比如) 30fps，那么一个剪辑有 60 帧。
    # 如果 `UniformTemporalSubsample(num_samples=2)`，那么会从这 60 帧中均匀采样 2 帧。
    # 这似乎是一个非常稀疏的采样，需要确认其意图。

    for video_path in video_paths: # 遍历输入的每个视频路径。
        video = EncodedVideo.from_path(
            video_path,              # 视频文件路径。
            decoder="decord",        # 指定使用 "decord" 解码器。
            decode_audio=False,      # 不解码音频部分。
            **{"sample_rate": sample_rate}, # 传递额外的解码参数，这里是音频采样率 (即使不解码音频)。
        )
        # 使用 EncodedVideo 类从路径加载视频。

        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)
        # 调用 get_clip_timepoints 函数，获取当前视频中所有时间剪辑的时间点。

        all_video = [] # 初始化一个列表，用于存储当前视频的所有处理后的剪辑帧。
        for clip_timepoints in all_clips_timepoints: # 遍历每个时间剪辑的时间点。
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            # 从 EncodedVideo 对象中获取指定时间段的剪辑数据。
            # 返回的 `clip` 是一个字典，通常包含 "video" 和 "audio" 键 (如果解码了音频)。
            if clip is None: # 如果获取剪辑失败。
                raise ValueError("No clip found") # 抛出错误。
            video_clip = frame_sampler(clip["video"])
            # 对提取到的视频帧数据 `clip["video"]` (通常是 [T, H, W, C] 或类似格式)
            # 应用 `frame_sampler` 进行时间帧的子采样。
            # `frame_sampler` 会返回一个 [num_samples, H, W, C] 或 [C, num_samples, H, W] 的张量，具体取决于实现。
            # PytorchVideo 通常输出 C, T, H, W。

            video_clip = video_clip / 255.0  # since this is float, need 0-1
            # 将视频帧的像素值从 [0, 255] (通常解码出来是 uint8) 缩放到 [0.0, 1.0] 范围，
            # 因为后续的 `NormalizeVideo` 和其他 PyTorch 操作通常期望浮点数输入。

            all_video.append(video_clip) # 将处理后的视频剪辑（一组帧）添加到列表中。

        all_video = [video_transform(clip) for clip in all_video]
        # 对列表中的每个视频剪辑（帧的集合）应用 `video_transform` (缩放和归一化)。
        # `video_transform` 期望输入是 C, T, H, W。

        all_video = SpatialCrop(224, num_crops=3)(all_video)
        # 创建一个 SpatialCrop 实例 (裁剪尺寸 224，生成 3 个空间裁剪)，
        # 并将其应用于 `all_video` 列表。
        # `all_video` 列表中的每个元素 (一个时间剪辑) 会被替换为 3 个空间裁剪后的版本。
        # 所以如果之前 `all_video` 有 N 个时间剪辑，现在它将有 N * 3 个元素。

        all_video = torch.stack(all_video, dim=0)
        # 将列表 `all_video` 中的所有视频剪辑（现在是经过时间采样、帧预处理、空间裁剪的）
        # 堆叠成一个新的张量。dim=0 表示在新的第0维度上堆叠。
        # 结果形状可能是 [num_total_clips, C, num_sampled_frames, H_cropped, W_cropped]。
        # 其中 num_total_clips = clips_per_video * num_spatial_crops。
        video_outputs.append(all_video) # 将当前视频的所有处理结果张量添加到 video_outputs 列表中。

    return torch.stack(video_outputs, dim=0).to(device)
    # 将 `video_outputs` 列表中的所有张量 (每个张量对应一个原始输入视频的处理结果) 再次堆叠，
    # 形成最终的批次输出张量，并在指定的 device 上。
    # 最终形状可能是 [batch_size, num_total_clips_per_video, C, num_sampled_frames, H_cropped, W_cropped]。