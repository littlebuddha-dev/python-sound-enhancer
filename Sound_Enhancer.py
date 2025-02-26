import librosa
import librosa.core
import soundfile as sf
import numpy as np
import torch
import os
import subprocess
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter, iirfilter, fftconvolve
from scipy.ndimage import gaussian_filter1d
from scipy import signal as scipy_signal
from typing import Tuple

# --- GPU サポートのチェック (Intel Mac での PyTorch MPS バックエンド) ---
USE_GPU = torch.backends.mps.is_available()
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" if USE_GPU else "1.0"


def to_gpu(array):
    """NumPy配列またはPyTorchテンソルをGPUに転送する関数"""
    if isinstance(array, torch.Tensor):
        return array.to("mps") if USE_GPU and array.device.type != "mps" else array  # 既にMPSデバイス上にある場合は何もしない
    return torch.tensor(array, dtype=torch.float32).to("mps") if USE_GPU else torch.tensor(array, dtype=torch.float32)

def to_cpu(tensor):
    """PyTorchテンソルをCPUに転送し、NumPy配列に変換する関数"""
    if USE_GPU and isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy() if tensor.device.type == "mps" else tensor.numpy() # CPU上に存在する場合は.numpy()のみ
    return tensor if isinstance(tensor, np.ndarray) else np.array(tensor)
    

# --- 統合されたエフェクト設定 ---
PARAMS = {
    # 【共通設定】
    "target_sr": 96000,            # ターゲットサンプリングレート
    "oversample_factor": 2,        # オーバーサンプリングファクター

    # 【BBE Processor設定】
    "bbe_on": True,                # BBE ProcessorのON/OFFフラグ
    "headroom_db": 3.0,            # ヘッドルーム (dB)

    "low_shelf_on": True,          # ローシェルフフィルター ON/OFF
    "low_cutoff_hz": 100,          # ローシェルフ カットオフ周波数
    "low_gain_db": 4.0,            # ローシェルフ ゲイン
    "low_q": 0.7,                  # ローシェルフ Q値

    "high_shelf_on": True,         # ハイシェルフフィルター ON/OFF
    "high_cutoff_hz": 6000,        # ハイシェルフ カットオフ周波数
    "high_gain_db": 5.0,           # ハイシェルフ ゲイン
    "high_q": 0.8,                 # ハイシェルフ Q値

    "bass_enhance_on": True,       # 低音強調 ON/OFF
    "bass_freq": 80,              # 低音強調 中心周波数
    "bass_intensity": 0.7,         # 低音強調 強度

    "exciter_on": True,           # Harmonic ExciterのON/OFFフラグ
    "exciter_amount": 0.5,        # エキサイター 量
    "exciter_low_freq": 3000,     # エキサイター 適用範囲 (低)
    "exciter_high_freq": 12000,    # エキサイター 適用範囲 (高)
    "preemph_strength": 0.8,         # プリエンファシス強度
    "hpss_harmonic_margin": 2.5,     # HPSSのマージン


    "dynamic_eq_on": True,        # ダイナミックEQ ON/OFF
    "dynamic_threshold": -24,     # ダイナミックEQ スレッショルド
    "dynamic_ratio": 1.5,          # ダイナミックEQ レシオ
    "dynamic_attack": 10,          # ダイナミックEQ アタックタイム
    "dynamic_release": 150,        # ダイナミックEQ リリースタイム
    "dynamic_bands": 3,            # ダイナミックEQ バンド数


    "spectral_processor_on": True,  # スペクトルプロセッサー ON/OFF
    "spectral_low": 1.2,           # スペクトルプロセッサー 低域強調
    "spectral_mid": 0.9,          # スペクトルプロセッサー 中域減衰
    "spectral_high": 1.3,         # スペクトルプロセッサー 高域強調
    "spectral_brightness": 1.1,   # スペクトルプロセッサー 明るさ

    "stereo_width_on": True,       # ステレオ幅拡張 ON/OFF
    "stereo_width": 1.3,           # ステレオ幅

    # 【コンプレッサー設定】
    "compressor_on": True,         # コンプレッサーのON/OFFフラグ
    "compression_ratio": 3.0,        # 全体のコンプレッション比
    "mid_compression_ratio": 1.8,    # ミッド信号のコンプレッション比
    "side_compression_ratio": 1.3,   # サイド信号のコンプレッション比

    # 【リミッター設定】
    "limiter_on": True,             # リミッターのON/OFFフラグ
    "limiter_threshold": 0.95,       # リミッタ閾値
    "output_gain": 1.0,              # 出力ゲイン

    # 【サチュレーション設定】
    "saturation_on": True,          # サチュレーションのON/OFFフラグ
    "saturation_amount": 0.5,       # 全体のサチュレーション量 (0-1)
    "saturation_drive": 1.0,        # 入力ゲイン (歪みやすさ)
    "saturation_mix": 1.0,          # ドライ/ウェット比率 (原音との混ざり具合)

    # トランジェントシェイパー設定
    "transient_shaper_on": True,
    "attack_gain": 1.5,
    "sustain_gain": 0.7,
    
    # ボーカル強調設定
    "vocal_presence_eq_on": True,
    "vocal_presence_center_freq": 3000,  # ボーカルの存在感を強調する中心周波数 (Hz)
    "vocal_presence_gain": 2.0,        # ゲイン (dB)
    "vocal_presence_q": 1.4,   
}


# --- ユーティリティ関数 ---

def normalize_audio(y: np.ndarray, headroom_db: float = 1.0) -> np.ndarray:
    """オーディオを正規化し、指定のヘッドルームを確保"""
    peak = np.max(np.abs(y))
    if peak > 0:
        target_peak = np.power(10, -headroom_db/20)
        return y * (target_peak / peak)
    return y


def butter_lowpass(cutoff, fs, order=4):
    """ローパスフィルター設計"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sos


def butter_highpass(cutoff, fs, order=4):
    """ハイパスフィルター設計"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', output='sos')
    return sos


def apply_lowpass_filter(data, cutoff, fs):
    """ローパスフィルター適用 (ゼロ位相遅延)"""
    sos = butter_lowpass(cutoff, fs)
    return sosfiltfilt(sos, data)


def apply_highpass_filter(data, cutoff, fs):
    """ハイパスフィルター適用 (ゼロ位相遅延)"""
    sos = butter_highpass(cutoff, fs)
    return sosfiltfilt(sos, data)


def design_shelf_filter(fc, gain_db, q_factor, fs, filter_type='low'):
    """シェルビングフィルタの設計 (バターワースフィルタを応用)"""
    nyq = 0.5 * fs
    wn = fc / nyq

    if filter_type == 'low':
        sos = butter(2, wn, btype='lowpass', output='sos')
        # ゲイン調整 (リニアスケール)
        gain = 10 ** (gain_db / 20)
        return sos, gain
    else:  # high shelf
        sos = butter(2, wn, btype='highpass', output='sos')
        # ゲイン調整 (リニアスケール)
        gain = 10 ** (gain_db / 20)
        return sos, gain

# --- エフェクター関数 ---


def harmonic_exciter(signal, amount, freq_range, sr):
    """高性能ハーモニックエキサイター：非線形歪みを高度に制御して高調波を生成

    Args:
        signal: 入力信号
        amount: エキサイター強度 (0.0～1.0)
        freq_range: エキサイター適用周波数範囲 [低域限界, 高域限界]
        sr: サンプリングレート

    Returns:
        処理済み信号
    """
    signal_cpu = to_cpu(signal)
    nyquist = sr / 2
    low_cutoff = freq_range[0] / nyquist
    high_cutoff = freq_range[1] / nyquist

    # 高度なスペクトル分割（クロスオーバーネットワーク）
    sos_low = scipy_signal.butter(
        8, low_cutoff, btype='highpass', output='sos')
    sos_high = scipy_signal.butter(
        8, high_cutoff, btype='lowpass', output='sos')

    # 非線形処理を施す帯域を分離（低域を保護）
    target_band = scipy_signal.sosfilt(sos_low, signal_cpu)
    target_band = scipy_signal.sosfilt(sos_high, target_band)

    # 元の信号から処理対象帯域を一時的に除去
    remaining_signal = signal_cpu - target_band

    # 高度なスペクトル解析のためのパラメータ設定
    if len(signal_cpu.shape) > 1:
        n_samples = signal_cpu.shape[1]
    else:
        n_samples = signal_cpu.shape[0]

    # 窓サイズを動的に調整
    window_size = min(4096, n_samples)

    # マルチステージ非線形処理（複数の歪み特性を組み合わせる）
    # 1. ソフトクリッピング（低次高調波生成）
    soft_clip = np.tanh(2.0 * amount * target_band)

    # 2. 波形整形（偶数次高調波強調）
    even_harm = target_band**2 * np.sign(target_band) * 0.4 * amount

    # 3. 非対称クリッピング（奇数次高調波生成）
    asym_factor = 0.2 * amount
    asym_clip = np.copy(target_band)
    pos_mask = (target_band > 0)
    neg_mask = (target_band < 0)
    asym_clip[pos_mask] = np.tanh(target_band[pos_mask] * (1 + asym_factor))
    asym_clip[neg_mask] = np.tanh(target_band[neg_mask] * (1 - asym_factor))

    # 4. 動的特性（入力レベルに応じた処理量調整）
    env = np.abs(target_band)
    env = scipy_signal.sosfilt(
        scipy_signal.butter(2, 30/(sr/2), btype='lowpass', output='sos'),
        env
    )
    dynamic_weight = 1.0 - 0.3 * np.clip(env, 0, 1)  # 大きな信号では処理を抑制

    # 5. 複合倍音生成
    combined = (0.6 * soft_clip + 0.15 * even_harm +
                0.25 * asym_clip) * dynamic_weight

    # 周波数依存処理を直接時間領域で実装
    # 高域を強調する簡易フィルタ
    high_emphasis = scipy_signal.sosfilt(
        scipy_signal.butter(
            2, freq_range[0] * 1.5 / nyquist, btype='highpass', output='sos'),
        combined
    )
    combined = combined * 0.7 + high_emphasis * 0.3

    # 位相整合処理
    # ヒルベルト変換を使用した位相調整はより単純な方法に置き換え
    # 最終的な混合前に全通過フィルタで位相を微調整
    allpass_sos = scipy_signal.butter(
        2, 1000 / nyquist, btype='highpass', output='sos')
    phase_aligned = scipy_signal.sosfilt(allpass_sos, combined)
    combined = combined * 0.85 + phase_aligned * 0.15

    # ダイナミックミックス（信号レベルに基づいて混合比を調整）
    # エンベロープフォロワーで信号レベルを検出
    envelope = np.abs(target_band)
    envelope = scipy_signal.sosfilt(
        scipy_signal.butter(2, 50/(sr/2), btype='lowpass', output='sos'),
        envelope
    )

    # レベルに応じてミックス比を調整（小さい信号ほど効果を強く）
    mix_ratio = amount * (1.0 - 0.3 * np.clip(envelope * 2, 0, 1))

    # 原音と処理音のミックス
    excited = target_band * (1 - mix_ratio) + combined * mix_ratio

    # 帯域制限して不要な成分を除去
    excited = scipy_signal.sosfilt(
        scipy_signal.butter(6, high_cutoff * 1.2,
                            btype='lowpass', output='sos'),
        excited
    )

    # 最終的な微調整（サチュレーションで音色を整える）
    final_drive = 1.0 + amount * 0.5
    excited = np.tanh(excited * final_drive) / final_drive

    # 複数の倍音生成技術を適用し、より豊かな高調波構造を作成
    # 1. 高次高調波を生成する波形整形
    high_harmonics = np.sign(excited) * np.abs(excited)**1.5 * 0.2 * amount

    # 2. 倍音のブレンド
    excited = excited * 0.85 + high_harmonics * 0.15

    # 3. 最終的な非線形処理（特に高域の明瞭度向上）
    clarity_enhance = np.tanh(
        excited * (1.0 + 0.5 * amount)) / (1.0 + 0.5 * amount)
    excited = excited * 0.7 + clarity_enhance * 0.3

    # スペクトルチルト補正（高域のバランスを整える）
    tilt_sos = scipy_signal.butter(
        2, 8000 / nyquist, btype='highpass', output='sos')
    tilt_correction = scipy_signal.sosfilt(tilt_sos, excited) * 0.3 * amount
    excited = excited - tilt_correction

    # 元の信号に処理済み帯域を合成
    result = remaining_signal + excited

    # クリッピング防止のための最終ゲイン調整
    peak = np.max(np.abs(result))
    if peak > 0.98:  # ピークが1に近い場合、わずかにゲインを下げる
        result = result * (0.98 / peak)

    return to_gpu(result)


def dynamic_eq(signal, sr, threshold_db=-20, ratio=2.0, attack_ms=5, release_ms=50, bands=3):
    """動的イコライザー：周波数帯域ごとにダイナミック処理"""
    signal_cpu = to_cpu(signal)
    result = np.copy(signal_cpu)
    threshold = 10 ** (threshold_db / 20)

    # 周波数帯域を分割
    band_edges = np.logspace(np.log10(80), np.log10(sr/2.5), bands+1)

    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)

    # attack_samples と release_samples が0にならないようにクリッピング
    attack_samples = max(1, attack_samples)
    release_samples = max(1, release_samples)

    for b in range(bands):
        # 各バンドのフィルタリング
        low_edge = band_edges[b] / (sr/2)
        high_edge = band_edges[b+1] / (sr/2)

        if b == 0:  # 低域
            sos = scipy_signal.butter(
                2, high_edge, btype='lowpass', output='sos')
        elif b == bands-1:  # 高域
            sos = scipy_signal.butter(
                2, low_edge, btype='highpass', output='sos')
        else:  # 中域
            sos = scipy_signal.butter(
                2, [low_edge, high_edge], btype='bandpass', output='sos')

        band_signal = scipy_signal.sosfilt(sos, signal_cpu)

        # 包絡線検出
        abs_signal = np.abs(band_signal)
        env = np.zeros_like(abs_signal)

        # 時定数を考慮した包絡線追従
        for i in range(1, len(env)):
            if abs_signal[i] > env[i-1]:  # アタック
                env[i] = abs_signal[i] + \
                    (env[i-1] - abs_signal[i]) * np.exp(-1.0/attack_samples)
            else:  # リリース
                env[i] = abs_signal[i] + \
                    (env[i-1] - abs_signal[i]) * np.exp(-1.0/release_samples)

        # コンプレッション/エンハンス処理
        gain = np.ones_like(env)
        mask = env > threshold
        gain[mask] = (threshold / env[mask]) ** (1 - 1/ratio)

        # スムージング
        gain = gaussian_filter1d(gain, sigma=sr/1000)

        # バンド信号に適用
        processed_band = band_signal * gain

        # 極端な値にならないようにクリッピング
        processed_band = np.clip(processed_band, -1.0, 1.0)  # 必要に応じて範囲を調整

        # 結果に加算
        result += processed_band * (0.5 + b * 0.25)  # 高域ほど強く
    return to_gpu(result)


def psychoacoustic_bass_enhancer(signal, sr, bass_freq=100, intensity=1.0):
    """心理音響的低音強調：基本周波数の倍音を加えて知覚的な低音を強化"""
    signal_cpu = to_cpu(signal)
    # 低域抽出
    nyquist = sr / 2
    cutoff = bass_freq / nyquist
    sos = scipy_signal.butter(4, cutoff, btype='lowpass', output='sos')
    bass = scipy_signal.sosfilt(sos, signal_cpu)

    # 二倍音生成（非線形処理）
    harmonic2 = np.sign(bass) * np.power(np.abs(bass), 0.5) * intensity * 0.6

    # 高調波フィルタリング（二倍音を整形）
    h2_cutoff = (bass_freq * 2) / nyquist
    sos_h2 = scipy_signal.butter(2, h2_cutoff, btype='lowpass', output='sos')
    harmonic2 = scipy_signal.sosfilt(sos_h2, harmonic2)

    # 元の信号とミックス
    return to_gpu(signal_cpu + harmonic2)


def spectral_processor(signal, sr, low_enhance=1.2, mid_cut=0.9, high_enhance=1.3, brightness=1.0):
    """スペクトル処理機：周波数特性を細かく調整"""
    signal_cpu = to_cpu(signal)

    # 短時間フーリエ変換
    nperseg = 2048
    noverlap = 1024

    # 信号長が nperseg より短い場合にパディング
    if signal_cpu.shape[-1] < nperseg:
        pad_width = nperseg - signal_cpu.shape[-1]
        if signal_cpu.ndim == 1:  # モノラル
            signal_cpu = np.pad(signal_cpu, (0, pad_width), mode='constant')
        elif signal_cpu.ndim == 2:  # ステレオ
            signal_cpu = np.pad(
                signal_cpu, ((0, 0), (0, pad_width)), mode='constant')

    f, t, Zxx = scipy_signal.stft(
        signal_cpu, sr, nperseg=nperseg, noverlap=noverlap)

    # 周波数帯域ごとに処理
    low_band = (f < 150)
    mid_band = (f >= 150) & (f < 4000)
    high_band = (f >= 4000)

    # ブールインデックスの長さを Zxx_mod の列数に合わせる
    low_band = low_band[:Zxx.shape[0]]
    mid_band = mid_band[:Zxx.shape[0]]
    high_band = high_band[:Zxx.shape[0]]

    # 各帯域の強調/減衰
    Zxx_mod = np.copy(Zxx)
    Zxx_mod[low_band, :] *= low_enhance
    Zxx_mod[mid_band, :] *= mid_cut
    Zxx_mod[high_band, :] *= high_enhance

    # 明るさ調整（高域の傾斜）
    if brightness != 1.0:
        brightness_curve = np.power(f / (sr/2), 2) * (brightness - 1) + 1
        brightness_curve = brightness_curve[:Zxx.shape[0]]  # 長さをZxxに合わせる
        Zxx_mod *= brightness_curve.reshape(-1, 1)  # ブロードキャスト可能なshapeに変形

    # 逆短時間フーリエ変換
    _, processed = scipy_signal.istft(
        Zxx_mod, sr, nperseg=nperseg, noverlap=noverlap)

    # 必要に応じて長さを調整
    if len(processed) > len(signal_cpu):
        processed = processed[:len(signal_cpu)]
    elif len(processed) < len(signal_cpu):
        processed = np.pad(processed, (0, len(signal_cpu) - len(processed)))

    return to_gpu(processed)


def multi_band_compressor(y, ratio):
    """マルチバンドコンプレッサー関数 (音圧を均一化)"""
    if not PARAMS["compressor_on"]:
        return y

    y_abs = torch.abs(y)
    gain_reduction = 1.0 / (1.0 + (ratio - 1.0) * y_abs)
    return y * gain_reduction


def encode_mid_side(left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mid/Side エンコーディング関数"""
    mid = ((left + right) / 2).astype(np.float32)
    side = ((left - right) / 2).astype(np.float32)
    return mid, side


def decode_mid_side(mid: np.ndarray, side: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mid/Side デコーディング関数"""
    left = (mid + side).astype(np.float32)
    right = (mid - side).astype(np.float32)
    return left, right


def stereo_enhancer(y):
    """ステレオエンハンサー関数 (ステレオ感を調整)"""
    if not PARAMS["stereo_width_on"] or len(y.shape) != 2:
        return y

    y_cpu = to_cpu(y)
    mid, side = encode_mid_side(y_cpu[0], y_cpu[1])

    mid_processed = to_cpu(multi_band_compressor(
        to_gpu(mid), PARAMS["mid_compression_ratio"]))
    side_processed = to_cpu(multi_band_compressor(
        to_gpu(side), PARAMS["side_compression_ratio"])) * PARAMS["stereo_width"]
    side_processed = apply_lowpass_filter(
        side_processed, 8000, PARAMS["target_sr"])

    left_out, right_out = decode_mid_side(mid_processed, side_processed)
    return to_gpu(np.stack([left_out, right_out]))


def limiter(y):
    """リミッター関数 (ピークレベルを制限)"""
    if not PARAMS["limiter_on"]:
        return y

    y_cpu = to_cpu(y)
    y_norm = normalize_audio(y_cpu, headroom_db=1.0)
    y_norm = y_norm * PARAMS["output_gain"]
    peak_level = np.max(np.abs(y_norm))

    if peak_level > PARAMS["limiter_threshold"]:
        y_norm = y_norm * (PARAMS["limiter_threshold"] / peak_level)
    return to_gpu(y_norm)


def diode_clipper(x, drive, amount):
    """ダイオードクリッピングを模倣したサチュレーション関数"""
    if not PARAMS["saturation_on"]:
        return x

    x = torch.tanh(x)
    x = x * drive

    pos = 0.5 * (torch.abs(x + amount) + (x + amount))
    neg = 0.5 * (torch.abs(x - amount) - (x - amount))
    y = torch.tanh(pos) - torch.tanh(neg)
    return y


def dynamic_saturation(y, sr):
    """入力信号の特性に応じてパラメータを動的に変化させるサチュレーション"""
    if not PARAMS["saturation_on"]:
        return y

    amount = PARAMS["saturation_amount"]
    drive = PARAMS["saturation_drive"]
    mix = PARAMS["saturation_mix"]

    y_cpu = to_cpu(y)
    frame_length = 2048
    hop_length = 512
    n_frames = 1 + (y_cpu.shape[1] - frame_length) // hop_length
    y_processed_length = (n_frames - 1) * hop_length + frame_length
    y_processed = np.zeros(
        (y_cpu.shape[0], y_processed_length), dtype=y_cpu.dtype)

    for i in range(y_cpu.shape[0]):
        rms = librosa.feature.rms(
            y=y_cpu[i], frame_length=frame_length, hop_length=hop_length)[0]
        rms_normalized = np.clip(rms / np.max(rms), 0.0, 1.0)

        dynamic_amount = amount + (rms_normalized * 0.2)
        dynamic_drive = drive + (rms_normalized * 1.0)

        frames = librosa.util.frame(
            y_cpu[i], frame_length=frame_length, hop_length=hop_length)

        for j, frame in enumerate(frames.T):
            saturated_frame = diode_clipper(
                to_gpu(frame), dynamic_drive[j], dynamic_amount[j])
            processed_frame = (1 - mix) * frame + mix * to_cpu(saturated_frame)
            y_processed[i, j * hop_length: j *
                        hop_length + frame_length] += processed_frame

    return to_gpu(y_processed)


def transient_shaper(y, attack_gain=1.2, sustain_gain=0.8, sr=44100):
    """簡易的なトランジェントシェイパー関数"""

    if not PARAMS["transient_shaper_on"]:
        return y

    y_cpu = to_cpu(y)
    y_processed = np.zeros_like(y_cpu)

    for i in range(y.shape[0]):
        # オンセット強度を計算
        onset_env = librosa.onset.onset_strength(y=y_cpu[i], sr=sr)
        # オンセットを検出 (閾値は調整が必要)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, backtrack=False)

        # フレームサイズとホップサイズ (librosa.stftのデフォルト値を使用)
        frame_length = 2048
        hop_length = 512

        # 各フレームにゲインを適用
        for frame_idx in range(len(onset_env)):
            start = frame_idx * hop_length
            end = min(start + frame_length, len(y_cpu[i]))

            # 現在のフレームがオンセットを含むかどうかを判定
            is_onset_frame = False
            for onset_sample in onsets:
                if start <= onset_sample < end:
                    is_onset_frame = True
                    break

            # ゲインを適用
            if is_onset_frame:
                y_processed[i, start:end] = y_cpu[i, start:end] * attack_gain
            else:
                y_processed[i, start:end] = y_cpu[i, start:end] * sustain_gain

    return to_gpu(y_processed)


def apply_reverb(signal, sr, impulse_response_path=None, wet_level=0.1, dry_level=0.9, pre_delay_ms=20):
    """
    リバーブを適用する関数。コンボリューションリバーブまたはシンプルなディレイベースのリバーブ。

    Args:
        signal: 入力信号 (NumPy array)
        sr: サンプリングレート
        impulse_response_path: インパルスレスポンスファイルのパス (コンボリューションリバーブの場合)。Noneの場合はシンプルなディレイベースリバーブを使用。
        wet_level: ウェット信号のレベル (0.0 - 1.0)
        dry_level: ドライ信号のレベル (0.0 - 1.0)
        pre_delay_ms: プリディレイ (ms)

    Returns:
        リバーブが適用された信号
    """
    signal_cpu = to_cpu(signal)

    if impulse_response_path:  # コンボリューションリバーブ
        ir, ir_sr = librosa.load(impulse_response_path, sr=sr, mono=False)
        ir = to_cpu(ir)

        # インパルスレスポンスがステレオの場合、入力信号もステレオであると仮定
        if ir.ndim == 2:
            if signal_cpu.ndim == 1:  # 入力がモノラルの場合、ステレオに拡張
                signal_cpu = np.stack([signal_cpu, signal_cpu], axis=0)
            reverbed_signal = np.stack([fftconvolve(signal_cpu[i], ir[i % ir.shape[0]], mode='same') for i in range(signal_cpu.shape[0])], axis=0)

        else: # インパルスレスポンスがモノラルの場合
            if signal_cpu.ndim == 2:  # 入力がステレオ
                reverbed_signal = np.stack([fftconvolve(signal_cpu[i], ir, mode='same') for i in range(signal_cpu.shape[0])], axis=0)
            else: # 入力がモノラル
                reverbed_signal = fftconvolve(signal_cpu, ir, mode='same')


    else:  # シンプルなディレイベースのリバーブ (簡易実装)
        pre_delay_samples = int(pre_delay_ms * sr / 1000)
        decay = 0.5  # 減衰率
        delay_samples = int(0.05 * sr)  # 50msのディレイ

        if signal_cpu.ndim == 2: # ステレオ
            reverbed_signal = np.zeros_like(signal_cpu)
            for i in range(signal_cpu.shape[0]):
              temp_signal = np.pad(signal_cpu[i], (pre_delay_samples, delay_samples * 5), mode='constant') # パディング
              for j in range(1, 6):
                  reverbed_signal[i] += np.roll(temp_signal, j * delay_samples)[:len(signal_cpu[i])] * (decay ** j) # ディレイと減衰

        else: #モノラル
            temp_signal = np.pad(signal_cpu, (pre_delay_samples, delay_samples * 5), mode='constant')
            reverbed_signal = np.zeros_like(temp_signal)
            for j in range(1, 6):
                reverbed_signal += np.roll(temp_signal, j * delay_samples) * (decay ** j) # ディレイと減衰
            reverbed_signal = reverbed_signal[:len(signal_cpu)]


    # ドライ/ウェットミックス
    if signal_cpu.ndim == 2:  # ステレオ
        return to_gpu(dry_level * signal_cpu + wet_level * reverbed_signal)
    else:
        return to_gpu(dry_level * signal_cpu + wet_level * reverbed_signal)

def design_peaking_filter(fc, gain_db, q_factor, fs):
    """ピーキングフィルタの設計"""
    nyq = 0.5 * fs
    w0 = fc / nyq
    alpha = np.sin(2 * np.pi * w0) / (2 * q_factor)

    b = [1 + alpha * (10**(gain_db/40)),
        -2 * np.cos(2 * np.pi * w0),
        1 - alpha * (10**(gain_db/40))]
    a = [1 + alpha / (10**(gain_db/40)),
        -2 * np.cos(2 * np.pi * w0),
        1 - alpha / (10**(gain_db/40))]

    return scipy_signal.tf2sos(b, a)
    
    
        
# --- メインBBEプロセッサー関数 (統合) ---
def advanced_bbe_processor(y, sr, params):
    """高度なBBEプロセッサー関数 (統合版)"""
    if not params["bbe_on"]:
        return y

    y_cpu = to_cpu(y)
    is_stereo = y_cpu.shape[0] == 2
    if not is_stereo:
        y_cpu = np.expand_dims(y_cpu, axis=0)  # モノラル処理のために調整
    y_processed = np.zeros_like(y_cpu)

    for i in range(y_cpu.shape[0]):
        channel = y_cpu[i]

        # NaN/inf チェック (入力)
        assert not np.any(np.isnan(channel)) and not np.any(
            np.isinf(channel)), "Input contains NaN or inf"

        # 1. 心理音響的低音強調
        if params["bass_enhance_on"]:
            channel = to_cpu(psychoacoustic_bass_enhancer(
                to_gpu(channel), sr, bass_freq=params["bass_freq"], intensity=params["bass_intensity"]
            ))
            assert not np.any(np.isnan(channel)) and not np.any(
                np.isinf(channel)), "psychoacoustic_bass_enhancer output contains NaN or inf"

       # 2. ローシェルフフィルタ
        if params["low_shelf_on"]:
            sos, gain = design_shelf_filter(
                params["low_cutoff_hz"], params["low_gain_db"], params["low_q"], sr, 'low'
            )
            channel = sosfilt(sos, channel)  # sosfilt を使用
            channel *= gain
            assert np.isfinite(channel).all(
            ), "low_shelf_filter output contains NaN or inf"

        # 3. ハイシェルフフィルタ
        if params["high_shelf_on"]:
            sos, gain = design_shelf_filter(
                params["high_cutoff_hz"], params["high_gain_db"], params["high_q"], sr, 'high'
            )
            channel = sosfilt(sos, channel)  # sosfiltを使用
            channel *= gain
            assert not np.any(np.isnan(channel)) and not np.any(
                np.isinf(channel)), "high_shelf_filter output contains NaN or inf"
        # 4. 倍音エキサイター
        if params["exciter_on"]:
            channel = to_cpu(harmonic_exciter(
                to_gpu(channel), params["exciter_amount"],
                [params["exciter_low_freq"], params["exciter_high_freq"]], sr
            ))
            assert not np.any(np.isnan(channel)) and not np.any(
                np.isinf(channel)), "harmonic_exciter output contains NaN or inf"

        # 5. 動的イコライザー
        if params["dynamic_eq_on"]:
            channel = to_cpu(dynamic_eq(
                to_gpu(channel), sr, threshold_db=params["dynamic_threshold"],
                ratio=params["dynamic_ratio"], attack_ms=params["dynamic_attack"],
                release_ms=params["dynamic_release"], bands=params["dynamic_bands"]
            ))
            assert not np.any(np.isnan(channel)) and not np.any(
                np.isinf(channel)), "dynamic_eq output contains NaN or inf"

        # 6. スペクトルプロセッサ
        if params["spectral_processor_on"]:
            channel = to_cpu(spectral_processor(
                to_gpu(channel), sr, low_enhance=params["spectral_low"],
                mid_cut=params["spectral_mid"], high_enhance=params["spectral_high"],
                brightness=params["spectral_brightness"]
            ))
            assert not np.any(np.isnan(channel)) and not np.any(
                np.isinf(channel)), "spectral_processor output contains NaN or inf"

        y_processed[i] = channel

    # 7. ステレオ幅エンハンス
    if is_stereo and params["stereo_width_on"]:
        y_processed[0], y_processed[1] = to_cpu(
            stereo_enhancer(to_gpu(y_processed)))
        assert not np.any(np.isnan(y_processed)) and not np.any(
            np.isinf(y_processed)), "stereo_enhancer output contains NaN or inf"

    # 8. トランジェントシェイパー
    y_processed = to_cpu(transient_shaper(to_gpu(y_processed),
                                          attack_gain=params["attack_gain"],
                                          sustain_gain=params["sustain_gain"],
                                          sr=sr))
    assert not np.any(np.isnan(y_processed)) and not np.any(
        np.isinf(y_processed)), "transient_shaper output contains NaN or inf"

    # 9. コンプレッサー
    y_processed = to_cpu(multi_band_compressor(
        to_gpu(y_processed), PARAMS["compression_ratio"]))
    assert not np.any(np.isnan(y_processed)) and not np.any(
        np.isinf(y_processed)), "multi_band_compressor output contains NaN or inf"

    # 10. サチュレーション
    y_processed = to_cpu(dynamic_saturation(to_gpu(y_processed), sr))
    assert not np.any(np.isnan(y_processed)) and not np.any(
        np.isinf(y_processed)), "dynamic_saturation output contains NaN or inf"

    # 11. 最終的な音量調整と歪み防止 (リミッター)
    y_processed = to_cpu(limiter(to_gpu(y_processed)))
    assert not np.any(np.isnan(y_processed)) and not np.any(
        np.isinf(y_processed)), "limiter output contains NaN or inf"

    # 12. ボーカル強調
    if params["vocal_presence_eq_on"]:
        sos = design_peaking_filter(params["vocal_presence_center_freq"], params["vocal_presence_gain"], params["vocal_presence_q"], sr)
        channel = sosfiltfilt(sos, channel)
    
    if not is_stereo:
        y_processed = y_processed.squeeze(0)  # モノラルの場合は次元を減らす
    return to_gpu(y_processed)


# --- エフェクト実行関数 ---
def effect_process(input_file: str, output_file: str):
    """メイン処理関数 (オーディオファイルにエフェクトを適用)"""
    print(f"[INFO] Processing: {input_file}")
    print("[INFO] Converting and loading audio file...")

    temp_wav = None
    try:
        if input_file.lower().endswith('.mp3'):
            temp_wav = input_file.replace(".mp3", "_temp.wav")
            subprocess.run(["ffmpeg", "-y", "-i", input_file, "-acodec", "pcm_s16le",
                            "-ar", str(PARAMS["target_sr"]), temp_wav], check=True)
            input_file = temp_wav

        y, sr = librosa.load(
            input_file, sr=PARAMS["target_sr"]//PARAMS["oversample_factor"], mono=False)
        print("[INFO] Audio file loaded. Applying oversampling...")
        y = to_gpu(y)

        # オーバーサンプリング
        y_os = to_gpu(librosa.core.resample(to_cpu(y), orig_sr=sr,
                      target_sr=PARAMS["target_sr"], res_type='kaiser_best'))

        # BBEプロセッシング
        y_processed = advanced_bbe_processor(y_os, PARAMS["target_sr"], PARAMS)

        # ダウンサンプリング
        y_hq = to_gpu(librosa.resample(to_cpu(
            y_processed), orig_sr=PARAMS["target_sr"], target_sr=PARAMS["target_sr"]//PARAMS["oversample_factor"]))

        sf.write(output_file, to_cpu(y_hq).T,
                 PARAMS["target_sr"]//PARAMS["oversample_factor"])

        if temp_wav:
            os.remove(temp_wav)

    except Exception as e:
        print(f"[ERROR] {e}")
        raise

    print("[INFO] Processing completed.")


# --- エフェクト実行 ---
if __name__ == '__main__':
    effect_process(
        "/Users/littlebuddha/Desktop/python/BBE/romance.mp3",  # 入力ファイル
        "/Users/littlebuddha/Desktop/python/BBE/romance_enhanced.wav"  # 出力ファイル
    )
