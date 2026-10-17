import os
import time
import math
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

import librosa
from sklearn.preprocessing import StandardScaler


class AcousticFeatureExtractor:
    """
    Extracts composite acoustic features with statistical pooling:
    - 13 MFCCs + 13 Delta MFCCs
    - Zero Crossing Rate (ZCR)
    - Spectral Centroid
    - Spectral Flux (Onset Strength)
    Pooled via [mean, std, max] to ensure duration invariance.
    """

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_features(self, audio_buffer: np.ndarray) -> np.ndarray:
        """
        Extracts pooled 1D composite feature vector from audio buffer.
        audio_buffer: 1D float32 array normalized to [-1.0, 1.0].
        """
        if len(audio_buffer) < 512:
            audio_buffer = np.pad(audio_buffer, (0, 512 - len(audio_buffer)))

        # 1. MFCCs + Deltas
        mfccs = librosa.feature.mfcc(
            y=audio_buffer,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=min(512, len(audio_buffer)),
            hop_length=128
        )
        delta_mfccs = librosa.feature.delta(mfccs)

        # 2. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio_buffer,
            frame_length=min(512, len(audio_buffer)),
            hop_length=128
        )

        # 3. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio_buffer,
            sr=self.sample_rate,
            n_fft=min(512, len(audio_buffer)),
            hop_length=128
        )

        # 4. Spectral Flux (Onset strength)
        flux = librosa.onset.onset_strength(
            y=audio_buffer,
            sr=self.sample_rate,
            n_fft=min(512, len(audio_buffer)),
            hop_length=128
        )
        flux = flux.reshape(1, -1)

        # Stack all feature contours: (n_features, time_steps)
        feature_matrix = np.vstack([mfccs, delta_mfccs, zcr, centroid, flux])

        # Statistical pooling: [mean, std, max] along time axis
        pooled_mean = np.mean(feature_matrix, axis=1)
        pooled_std = np.std(feature_matrix, axis=1)
        pooled_max = np.max(feature_matrix, axis=1)

        feature_vector = np.concatenate([pooled_mean, pooled_std, pooled_max])
        return feature_vector.astype(np.float64)


class AcousticClassifier:
    """
    Acoustic Intent Classifier featuring:
    - Dual-mechanism noise floor (Leaky EMA + 1.5s timeout latch)
    - Frozen calibration scaler
    - Data-derived R_max cluster radius
    - Margin-checked nearest-neighbor against supra-threshold negative class
    - Safe dynamic memory bank with immutable anchors
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        snr_multiplier: float = 3.0,
        margin: float = 0.8,
        alpha_down: float = 0.05,
        max_sustained_sec: float = 1.5,
        max_dynamic_samples: int = 20,
    ):
        self.sample_rate = sample_rate
        self.snr_multiplier = snr_multiplier
        self.margin = margin
        self.alpha_down = alpha_down
        self.max_sustained_sec = max_sustained_sec
        self.max_dynamic_samples = max_dynamic_samples

        self.extractor = AcousticFeatureExtractor(sample_rate=sample_rate)

        # Noise floor tracking state
        self.noise_floor: float = 0.005
        self.current_gate: float = 0.015
        self.sustained_frames: int = 0
        self.max_sustained_frames: int = int(max_sustained_sec * (sample_rate / 512))

        # Model storage
        self.scaler: Optional[StandardScaler] = None
        self.anchor_clicks: List[np.ndarray] = []     # Immutable calibration anchors
        self.dynamic_clicks: List[np.ndarray] = []    # Rolling dynamic updates
        self.noise_bank: List[np.ndarray] = []        # Supra-threshold negative samples
        self.r_max: float = 5.0                       # Derived from calibration dispersion

        self.is_calibrated: bool = False

    def update_noise_floor(self, rms: float) -> bool:
        """
        Dual-mechanism noise floor update.
        Returns True if signal is above gate (potential intent sound), False if background.
        """
        if rms >= self.current_gate:
            self.sustained_frames += 1
            if self.sustained_frames > self.max_sustained_frames:
                # Timeout latch: persistent noise (fan, TV) detected, re-latch floor
                self.noise_floor = rms
                self.current_gate = self.noise_floor * self.snr_multiplier
                self.sustained_frames = 0
            return True
        else:
            self.sustained_frames = 0
            # Quiet: smooth downward adaptation
            self.noise_floor = (1.0 - self.alpha_down) * self.noise_floor + self.alpha_down * rms
            self.current_gate = max(0.002, self.noise_floor * self.snr_multiplier)
            return False

    def calibrate(
        self,
        raw_click_audios: List[np.ndarray],
        raw_negative_audios: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Calibrate classifier with recorded click and supra-threshold negative audios.
        Fits and freezes the StandardScaler and computes data-derived R_max.
        """
        click_feats = [self.extractor.extract_features(a) for a in raw_click_audios]
        noise_feats = [self.extractor.extract_features(a) for a in raw_negative_audios]

        if len(click_feats) < 3 or len(noise_feats) < 3:
            raise ValueError("Need at least 3 click samples and 3 noise samples for calibration.")

        all_feats = np.vstack(click_feats + noise_feats)

        # Fit and freeze scaler
        self.scaler = StandardScaler()
        self.scaler.fit(all_feats)

        # Scale features
        scaled_clicks = [self.scaler.transform(f.reshape(1, -1))[0] for f in click_feats]
        scaled_noises = [self.scaler.transform(f.reshape(1, -1))[0] for f in noise_feats]

        self.anchor_clicks = scaled_clicks
        self.dynamic_clicks = []
        self.noise_bank = scaled_noises

        # Compute empirical R_max from leave-one-out nearest-neighbor within clicks
        loo_distances = []
        for i, c_i in enumerate(scaled_clicks):
            other_clicks = [c for j, c in enumerate(scaled_clicks) if j != i]
            min_dist = min(np.linalg.norm(c_i - c_j) for c_j in other_clicks)
            loo_distances.append(min_dist)

        # 95th percentile * 1.35 margin factor
        self.r_max = float(np.percentile(loo_distances, 95) * 1.35)
        self.is_calibrated = True

        # Compute Leave-One-Out (LOO) Confusion Matrix
        loo_eval = self._evaluate_leave_one_out(scaled_clicks, scaled_noises)

        return {
            "r_max": self.r_max,
            "mean_intra_click_dist": float(np.mean(loo_distances)),
            "loo_evaluation": loo_eval,
            "click_samples_count": len(scaled_clicks),
            "noise_samples_count": len(scaled_noises),
        }

    def _evaluate_leave_one_out(
        self, clicks: List[np.ndarray], noises: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Honest Leave-One-Out validation on the calibration data."""
        true_positives = 0
        false_negatives = 0
        for i, c_i in enumerate(clicks):
            other_clicks = [c for j, c in enumerate(clicks) if j != i]
            d_click = min(np.linalg.norm(c_i - c) for c in other_clicks)
            d_noise = min(np.linalg.norm(c_i - n) for n in noises)
            if d_click <= self.r_max and (d_click < d_noise - self.margin):
                true_positives += 1
            else:
                false_negatives += 1

        true_negatives = 0
        false_positives = 0
        for i, n_i in enumerate(noises):
            other_noises = [n for j, n in enumerate(noises) if j != i]
            d_click = min(np.linalg.norm(n_i - c) for c in clicks)
            d_noise = min(np.linalg.norm(n_i - n) for n in other_noises)
            if d_click <= self.r_max and (d_click < d_noise - self.margin):
                false_positives += 1
            else:
                true_negatives += 1

        total = len(clicks) + len(noises)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "tp": true_positives,
            "fn": false_negatives,
            "tn": true_negatives,
            "fp": false_positives,
            "confusion_matrix": [
                [true_positives, false_negatives],
                [false_positives, true_negatives],
            ]
        }

    def classify_audio(self, raw_audio: np.ndarray) -> Tuple[bool, float, float]:
        """
        Classifies incoming candidate sound buffer.
        Returns: (is_click, confidence_score, d_click)
        """
        if not self.is_calibrated or self.scaler is None:
            return False, 0.0, float("inf")

        feat = self.extractor.extract_features(raw_audio)
        # Transform using FROZEN scaler
        z = self.scaler.transform(feat.reshape(1, -1))[0]

        # All available click samples (immutable anchors + dynamic)
        all_clicks = self.anchor_clicks + self.dynamic_clicks
        d_click = min(float(np.linalg.norm(z - c)) for c in all_clicks)
        d_noise = min(float(np.linalg.norm(z - n)) for n in self.noise_bank)

        # Decision rule: inside R_max and separated from noise by margin
        is_click = (d_click <= self.r_max) and (d_click < d_noise - self.margin)

        # Confidence: inverse distance mapping
        confidence = max(0.0, min(1.0, 1.0 - (d_click / max(self.r_max, 1e-5))))

        return is_click, float(confidence), float(d_click)

    def add_verified_sample(self, raw_audio: np.ndarray):
        """
        Continuous Learning: Adds user-verified / non-undone click sample.
        Transformed with frozen scaler; dynamic bank capped to max_dynamic_samples.
        """
        if not self.is_calibrated or self.scaler is None:
            return
        feat = self.extractor.extract_features(raw_audio)
        z = self.scaler.transform(feat.reshape(1, -1))[0]
        self.dynamic_clicks.append(z)
        if len(self.dynamic_clicks) > self.max_dynamic_samples:
            self.dynamic_clicks.pop(0)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "anchor_clicks": self.anchor_clicks,
            "dynamic_clicks": self.dynamic_clicks,
            "noise_bank": self.noise_bank,
            "r_max": self.r_max,
            "noise_floor": self.noise_floor,
            "current_gate": self.current_gate,
            "margin": self.margin,
            "sample_rate": self.sample_rate,
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.anchor_clicks = data["anchor_clicks"]
        self.dynamic_clicks = data["dynamic_clicks"]
        self.noise_bank = data["noise_bank"]
        self.r_max = data["r_max"]
        self.noise_floor = data.get("noise_floor", 0.005)
        self.current_gate = data.get("current_gate", 0.015)
        self.margin = data.get("margin", 0.8)
        self.sample_rate = data.get("sample_rate", 16000)
        self.is_calibrated = True
