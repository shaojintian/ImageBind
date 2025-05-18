from datasets import load_dataset, Image, Audio, DatasetDict, concatenate_datasets, interleave_datasets,Dataset,Features,Sequence, Value
from transformers import AutoTokenizer
import torch
import torchaudio
import torchvision.transforms as T
from PIL import Image as PILImage
import numpy as np
from decord import VideoReader, cpu
import os
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("./dataset/preprocess.log"))

#ouput tensor format
# 假设你的数据存储路径
# 对于 LAION-5B 这样的超大规模数据集，通常不会直接完整下载，而是使用其索引或子集。
# 这里我们用占位符或更小的数据集作为示例。

# --- 图像 (HWC) ---
# 假设 LAION-5B 的一个子集图像存储在 "path/to/laion_images"
# 或者使用一个标准图像数据集作为例子，比如 'cifar10' 或 'imagefolder'
# image_dataset = load_dataset("imagefolder", data_dir="path/to/laion_image_subset")
# 为了演示，我们用一个Hugging Face上的小数据集
try:
    image_dataset = load_dataset("beans", split='train[:1%]',trust_remote_code=True,cache_dir="./dataset/cache") # 使用一个小的图像数据集的1%作为示例
except Exception as e:
    print(f"Could not load 'beans' dataset, using dummy image data: {e}")
    # 创建一个虚拟的图像数据集作为后备
    dummy_image_paths = ["dummy_image1.png", "dummy_image2.png"]
    for p in dummy_image_paths: # 创建虚拟图片
        if not os.path.exists(p): PILImage.new('RGB', (60, 30), color = 'red').save(p)
    image_dataset = load_dataset("imagefolder", data_files={"train": dummy_image_paths})['train']


# --- 文本 (L or LD) ---
# 对于 LAION-5B 的文本部分或 The Pile
# text_dataset = load_dataset("text", data_files={"train": "path/to/laion_text_subset.txt"})
# 为了演示，我们用一个Hugging Face上的小数据集
try:
    text_dataset = load_dataset("glue", "mrpc", split='train[:1%]',cache_dir="./dataset/cache") # 使用一个小的文本数据集的1%作为示例
    # 我们只关心文本本身，例如'sentence1'字段
except Exception as e:
    print(f"Could not load 'glue' dataset, using dummy text data: {e}")
    dummy_texts = ["This is a sample sentence.", "Another example of text data."]
    text_dataset = Dataset.from_dict({"sentence1": dummy_texts})


# --- 视频 (THWC) ---
# WebVid 数据集通常提供 URL 或需要专门的下载脚本。
# 假设我们有一个包含视频文件路径的元数据文件，或者视频直接在文件夹中。
# 为了演示，我们将创建一个虚拟的视频文件和数据集
video_dir = "HuggingFaceFV/finevideo"
try:
    # load_dataset 会自动将文件夹名作为标签
    # 它期望的是一个包含 train/test/validation 等子目录的结构，
    # 或者直接是类别子文件夹（这种情况下会全部加载到一个 'train' split 中）
    # 如果 HuggingFaceFV/finevideo 直接包含类别子文件夹：
    video_dataset = load_dataset(
        "HuggingFaceFV/finevideo",
        trust_remote_code=True,
        split="train",
        streaming=True,
        cache_dir="./dataset/cache"
        # drop_labels=False, # 默认会加载标签
    )
    # 这个加载器主要设计用于图像，对于视频，它会加载文件路径。
    # 你仍然需要后续的 .map() 操作来实际处理视频内容。

    print("Dataset loaded using 'videofolder':")
    print(video_dataset) # 通常会得到一个 DatasetDict, e.g., {'train': Dataset(...)}

    # 注意: 'videofolder' 默认将文件路径存储在 'video' 键下。
    # 如果你想在加载时就指定视频内容应该如何被“看待”（即使实际加载在 map 中进行），
    # 你可能需要自定义加载脚本，或者在加载后使用 .cast_column()，但这比较复杂。
    # 通常的做法是先加载文件路径，然后在 .map() 中处理。

except Exception as e:
    print(f"Error loading dataset with 'videofolder': {e}")
    # print("Make sure your directory structure is correct for 'videofolder' (e.g., class subdirectories).")
    # print("The 'videofolder' builder might be more oriented towards images by default.")
    # print("For video, you might need to use it to get file paths and then map a video loading function.")





# --- 音频 (Spectrograms - TFD) ---
# AudioSet 通常也很大，提供 YouTube ID 或已提取的特征。
# 假设我们有一个包含音频文件路径的元数据文件。
# 为了演示，我们用一个Hugging Face上的小音频数据集
try:
    audio_dataset = load_dataset("mozilla-foundation/common_voice_17_0", "ab",split='train',trust_remote_code=True,cache_dir="./dataset/cache") # 包含音频路径
except Exception as e:
    print(f"Could not load 'voice' dataset, using dummy audio data: {e}")
    # 创建虚拟音频文件和数据集 (更复杂，通常会用现有文件)
    # 为了简单，这里只创建元数据
    dummy_audio_data = [{"path": "dummy1.wav", "sentence": "dummy"}, {"path": "dummy2.wav", "sentence": "dummy"}]
    # 你需要手动创建这些 .wav 文件，或者使用 torchaudio 生成简单的波形并保存
    # if not os.path.exists("dummy1.wav"):
    #     sample_rate = 16000
    #     dummy_waveform = torch.sin(2 * torch.pi * torch.arange(0, 1, 1/sample_rate) * 440) # 1 sec 440Hz sine
    #     torchaudio.save("dummy1.wav", dummy_waveform.unsqueeze(0), sample_rate)
    #     torchaudio.save("dummy2.wav", dummy_waveform.unsqueeze(0), sample_rate)

    if os.path.exists("dummy1.wav") and os.path.exists("dummy2.wav"):
        audio_dataset = Dataset.from_list(dummy_audio_data)
    else:
        print("Dummy audio files not found. Audio processing will be skipped or use placeholder logic.")
        audio_dataset = Dataset.from_list([{"path": "placeholder.wav", "sentence": "dummy"}]*2)


# --- 预处理器 ---

# 1. 图像预处理 (HWC)
IMG_SIZE = (224, 224) # OmniTensor 的 UTT 可以处理不同大小，但通常会归一化到一个范围
image_transform = T.Compose([
    T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x), # 确保是RGB
    T.Resize(IMG_SIZE),
    T.ToTensor(), # Converts to CHW and scales to [0, 1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 标准化
    T.Lambda(lambda x: x.permute(1, 2, 0)) # CHW -> HWC
])

def preprocess_image(examples):
    # 'image' 字段通常是一个 PIL Image 对象
    examples['image_tensor'] = [image_transform(image) for image in examples['image']]
    return examples

# 2. 文本预处理 (L for Token IDs)
# OmniTensor的UTT可以处理Token ID序列，然后内部进行embedding和patchification
TOKENIZER_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
MAX_LENGTH = 77 # 根据模型调整

def preprocess_text(examples):
    # 假设文本在 'sentence1' 字段 (如 MRPC 数据集)
    # 如果是纯文本文件加载的，字段名可能是 'text'
    text_inputs = examples.get('sentence1', examples.get('text'))
    if text_inputs is None:
        raise ValueError("Text field not found in examples. Expected 'sentence1' or 'text'.")

    tokenized_outputs = tokenizer(
        text_inputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt" # 返回 PyTorch 张量
    )
    examples['text_ids'] = tokenized_outputs.input_ids # Shape: (batch_size, L)
    examples['text_attention_mask'] = tokenized_outputs.attention_mask
    return examples

# 3. 视频预处理 (THWC)
NUM_FRAMES = 8  # T
VIDEO_IMG_SIZE = (112, 112) # 视频帧通常比静态图像小

video_frame_transform = T.Compose([
    T.ToPILImage(), # decord 输出 numpy, 转 PIL
    T.Resize(VIDEO_IMG_SIZE),
    T.ToTensor(), # CHW, [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Lambda(lambda x: x.permute(1, 2, 0)) # CHW -> HWC
])

def sample_frames(video_path, num_frames):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0: return [] # 空视频
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy() # (T, H, W, C)
        return frames
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        # 返回占位符或空列表
        # OmniTensor 的 UTT 需要能处理这种情况，比如忽略这个样本或用一个特殊的空tensor
        return np.zeros((num_frames, VIDEO_IMG_SIZE[0], VIDEO_IMG_SIZE[1], 3), dtype=np.float32)


def preprocess_video(examples):
    video_tensors = []
    for video_path in examples['video_path']:
        if "placeholder.mp4" in video_path: # 处理占位符
             frames_tensor = torch.zeros((NUM_FRAMES, VIDEO_IMG_SIZE[0], VIDEO_IMG_SIZE[1], 3))
        else:
            raw_frames = sample_frames(video_path, NUM_FRAMES) # T, H, W, C (numpy)
            if len(raw_frames) == 0: # 如果采样失败
                frames_tensor = torch.zeros((NUM_FRAMES, VIDEO_IMG_SIZE[0], VIDEO_IMG_SIZE[1], 3))
            else:
                # (T,H,W,C) -> (T,C,H,W) for transform -> (T,H,W,C)
                processed_frames = [video_frame_transform(frame) for frame in raw_frames] # List of (H,W,C) tensors
                frames_tensor = torch.stack(processed_frames) # THWC tensor
        video_tensors.append(frames_tensor)
    examples['video_tensor'] = video_tensors
    return examples

# 4. 音频预处理 (Spectrograms - TFD)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64 # F dimension
MAX_AUDIO_FRAMES = 1024 # T dimension (裁剪或填充频谱图的时间轴)

mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)
amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB()

def preprocess_audio(examples):
    audio_tensors = []
    # 'common_language' 数据集的音频在 'audio' 字段下，是一个包含 'path' 和 'array' 的dict
    # 如果是其他数据集，可能直接是路径字符串
    for audio_item in examples['audio']:
        audio_path = audio_item.get('path', None)
        if "placeholder.wav" in audio_path if audio_path else True: # 处理占位符
            mel_spec_db = torch.zeros((N_MELS, MAX_AUDIO_FRAMES)) # FD, (T will be added)
        elif audio_path and os.path.exists(audio_path):
            try:
                waveform, sr = torchaudio.load(audio_path)
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
                if waveform.shape[0] > 1: # 转为单声道
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                mel_spec = mel_spectrogram_transform(waveform) # (Channels, F, T_orig) e.g. (1, 64, T_orig)
                mel_spec_db = amplitude_to_db_transform(mel_spec.squeeze(0)) # (F, T_orig)

                # 裁剪或填充时间轴
                if mel_spec_db.shape[1] > MAX_AUDIO_FRAMES:
                    mel_spec_db = mel_spec_db[:, :MAX_AUDIO_FRAMES]
                else:
                    padding = MAX_AUDIO_FRAMES - mel_spec_db.shape[1]
                    mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding), mode='constant', value=0)
            except Exception as e:
                print(f"Error processing audio {audio_path}: {e}")
                mel_spec_db = torch.zeros((N_MELS, MAX_AUDIO_FRAMES))
        else: # 如果没有路径或文件不存在
             mel_spec_db = torch.zeros((N_MELS, MAX_AUDIO_FRAMES))

        # (F, T) -> (T, F, D=1) for OmniTensor (TFD)
        audio_tensors.append(mel_spec_db.permute(1, 0).unsqueeze(-1))

    examples['audio_spectrogram_tensor'] = audio_tensors
    return examples


# --- 应用预处理 ---
# 注意：对于非常大的数据集，`map` 操作会创建缓存。确保有足够磁盘空间。
# `num_proc` 可以加速处理，但也会增加内存消耗。
# `remove_columns` 可以删除不再需要的原始列。

print("Preprocessing images...")
processed_image_dataset = image_dataset.map(
    preprocess_image,
    batched=True,
    #s=Features({'image': Image(), 'image_tensor': Sequence(feature=Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None), length=-1, id=None)})
    # remove_columns=['image', 'label'] # 保留image_tensorfeature
)
processed_image_dataset.set_format(type='torch', columns=['image_tensor'])
processed_image_dataset.save_to_disk("./dataset/processed_image_dataset")


print("Preprocessing text...")
# 为文本数据集定义特征 (如果需要严格控制)
# text_features = Features({
#     'sentence1': Value('string'), 'sentence2': Value('string'), 'label': ClassLabel(num_classes=2), 'idx': Value('int32'),
#     'text_ids': Sequence(Value('int64')), 'text_attention_mask': Sequence(Value('int64'))
# })
processed_text_dataset = text_dataset.map(
    preprocess_text,
    batched=True,
    # features=text_features # 如果GLUE数据集，它自带features
    # remove_columns=['sentence1', 'sentence2', 'label', 'idx'] # 保留 text_ids, text_attention_mask
)
processed_text_dataset.set_format(type='torch', columns=['text_ids', 'text_attention_mask'])
processed_text_dataset.save_to_disk("./dataset/processed_text_dataset")

print("Preprocessing video...")
# video_features = Features({
#     'video_path': Value('string'),
#     'video_tensor': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='float32'), length=3), length=VIDEO_IMG_SIZE[1]), length=VIDEO_IMG_SIZE[0], dimensions=(NUM_FRAMES, VIDEO_IMG_SIZE[0], VIDEO_IMG_SIZE[1], 3))
# })
if "placeholder.mp4" not in video_dataset[0]['video_path']: # 仅当有真实视频时处理
    processed_video_dataset = video_dataset.map(
        preprocess_video,
        batched=True,
        # features=video_features # 定义特征可以帮助调试和类型检查
        # remove_columns=['video_path']
    )
    processed_video_dataset.set_format(type='torch', columns=['video_tensor'])
    processed_video_dataset.save_to_disk("./dataset/processed_video_dataset")
else:
    print("Skipping video processing due to placeholder files.")
    processed_video_dataset = video_dataset # 或者创建一个包含零张量的数据集


print("Preprocessing audio...")
# audio_features = Features({
#     'audio': Audio(sampling_rate=SAMPLE_RATE), # 假设原始数据包含Audio特性
#     'sentence': Value('string'), # 假设原始数据有这个字段
#     'audio_spectrogram_tensor': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='float32'), length=1), length=N_MELS), length=MAX_AUDIO_FRAMES) # T, F, D=1
# })
if "placeholder.wav" not in audio_dataset[0]['path'] if 'path' in audio_dataset[0] else True:
    processed_audio_dataset = audio_dataset.map(
        preprocess_audio,
        batched=True,
        # features=audio_features,
        # remove_columns=['audio', 'sentence', 'path'] # 保留 audio_spectrogram_tensor
    )
    processed_audio_dataset.set_format(type='torch', columns=['audio_spectrogram_tensor'])
    processed_audio_dataset.save_to_disk("./dataset/processed_audio_dataset")
else:
    print("Skipping audio processing due to placeholder files.")
    processed_audio_dataset = audio_dataset # 或创建零张量数据集

print("\n--- Sample preprocessed outputs ---")
print(f"Sample image tensor shape: {processed_image_dataset[0]['image_tensor'].shape}") # H, W, C
logger.info(f"Sample text_ids shape: {processed_text_dataset[0]['text_ids'].shape}") # L
if "placeholder.mp4" not in video_dataset[0]['video_path'] and hasattr(processed_video_dataset, '__getitem__') and 'video_tensor' in processed_video_dataset[0]:
    print(f"Sample video tensor shape: {processed_video_dataset[0]['video_tensor'].shape}") # T, H, W, C
else:
    print("Sample video tensor: Not processed or placeholder.")
if "placeholder.wav" not in audio_dataset[0]['path'] if 'path' in audio_dataset[0] else True and hasattr(processed_audio_dataset, '__getitem__') and 'audio_spectrogram_tensor' in processed_audio_dataset[0]:
    print(f"Sample audio spectrogram tensor shape: {processed_audio_dataset[0]['audio_spectrogram_tensor'].shape}") # T, F, D=1
else:
    print("Sample audio spectrogram tensor: Not processed or placeholder.")


# --- 组合数据集 (可选) ---
# 如果你要用一个 DataLoader 同时加载不同模态的数据（例如，用于 M3 预训练，其中每个样本是一种模态）
# 你可能需要将它们合并。 `interleave_datasets` 是一种常见方式，可以按比例混合。
# 注意：每个数据集的列名需要统一，或者在 collate_fn 中特殊处理。
# 为了让 OmniTensor 处理，每个样本应该是一个包含特定模态张量的字典。

# 创建一个 DatasetDict，模拟多模态场景
# 注意：OmniTensor 论文中 M3 预训练是针对单模态输入进行 Mask然后重建。
# 所以，数据加载器可能一次只提供一种模态的张量，或者一个样本包含多种模态，然后模型选择一种处理。
# 如果一次只处理一种，那么可以分别创建 DataLoader。
# 如果要混合，需要小心处理列名和数据结构。

# 示例：创建一个包含所有已处理数据集的列表
# all_processed_datasets = []
# if processed_image_dataset: all_processed_datasets.append(processed_image_dataset.rename_column("image_tensor", "input_tensor"))
# if processed_text_dataset: all_processed_datasets.append(processed_text_dataset.rename_column("text_ids", "input_tensor"))
# ... 等等，需要确保 "input_tensor" 的数据类型和结构被 OmniTensor 的 UTT 接受。
# 或者，DataLoader 的 collate_fn 可以做得更智能，将不同key的tensor包装成 OmniTensor期望的输入格式。

# 例如，一个更简单的 DataLoader 使用：
from torch.utils.data import DataLoader

def identity_collate(batch):
    # 如果batch中的每个元素都已经是准备好的tensor或tensor dict
    # 并且 OmniTensor 的 UTT 可以处理这种 list of dicts (每个dict含一个tensor)
    return batch # 或者根据 OmniTensor 的具体输入要求调整

# image_loader = DataLoader(processed_image_dataset, batch_size=4, collate_fn=identity_collate)
# text_loader = DataLoader(processed_text_dataset, batch_size=4, collate_fn=identity_collate)
# ...

# for batch in image_loader:
#     # batch 会是一个 list of dicts, e.g., [{'image_tensor': tensor1}, {'image_tensor': tensor2}, ...]
#     # 或者，如果 set_format 后，它可能是 dict of lists/tensors: {'image_tensor': batched_tensor}
#     # 这取决于你的具体 set_format 和 DataLoader 行为。
#     # 假设 batch = {'image_tensor': batched_hwc_tensor}
#     hwc_tensors = batch['image_tensor']
#     # 送入 OmniTensor (伪代码)
#     # output = omni_tensor_model([hwc_tensors]) # OmniTensor可能期望一个tensor列表
#     print("Image batch shape:", hwc_tensors.shape)
#     break