import io
import wave
import pygame
from piper import PiperVoice

# 全局TTS实例，避免重复初始化
tts_instance = None

def init_tts(language="zh"):
    """初始化TTS引擎（全局只初始化一次）"""
    global tts_instance
    if tts_instance is not None:
        return tts_instance
    
    HOME = "/home/jetson"
    try:
        # 加载中文模型（适配Jetson路径）
        if language == 'zh':
            tts_model = f"{HOME}/MODELS/tts/zh/zh_CN-huayan-medium.onnx"
            tts_json = f"{HOME}/MODELS/tts/zh/zh_CN-huayan-medium.onnx.json"
        else:
            tts_model = f"{HOME}/MODELS/tts/en/en_US-libritts-high.onnx"
            tts_json = f"{HOME}/MODELS/tts/en/en_US-libritts-high.onnx.json"
        
        # 初始化Piper TTS
        synthesizer = PiperVoice.load(tts_model, tts_json)
        synthesizer.volume = 2.0  # 音量
        synthesizer.length_scale = 1.0  # 语速
        tts_instance = synthesizer
        print("Piper TTS初始化成功")
        return synthesizer
    except Exception as e:
        print(f"Piper TTS初始化失败: {e}")
        return None

def synthesize_and_play(text):
    """
    核心接口：传入文字，直接合成并播放音频（无需保存文件）
    :param text: 要播放的文字内容
    """
    # 1. 初始化TTS（首次调用自动初始化）
    synthesizer = init_tts()
    if synthesizer is None:
        print(f"TTS未初始化，跳过语音播放：{text}")
        return
    
    # 2. 合成音频到内存字节流
    try:
        audio_bytes = io.BytesIO()
        with wave.open(audio_bytes, "wb") as wav_file:
            wav_file.setnchannels(1)    # 单声道
            wav_file.setsampwidth(2)    # 16位采样
            wav_file.setframerate(22050)# 采样率
            wav_file.setcomptype('NONE', 'NONE')
            synthesizer.synthesize_wav(text, wav_file)
        audio_bytes.seek(0)  # 重置指针
    except Exception as e:
        print(f"音频合成失败：{e}，文字：{text}")
        return
    
    # 3. 播放内存中的音频
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes)
        pygame.mixer.music.play()
        # 等待播放完成
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"音频播放失败：{e}")
    finally:
        pygame.mixer.quit()

# 测试代码
if __name__ == "__main__":
    synthesize_and_play("正在初始化位置")
    synthesize_and_play("已到达目标点1.00，2.00")