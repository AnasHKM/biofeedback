import sys
import time
from pathlib import Path
from collections import deque

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QGraphicsColorizeEffect
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QColor, QPainter

import numpy as np
import pyqtgraph as pg
from pyfirmata import Arduino, util


class BreathingGuideWidget(QWidget):
    """
    A circle that expands (inhale), holds, then shrinks (exhale).
    """
    def __init__(self, parent=None, inhale_time=4.0, hold_time=1.0, exhale_time=6.0):
        super().__init__(parent)
        self.inhale_time = inhale_time
        self.hold_time = hold_time
        self.exhale_time = exhale_time
        self.cycle_time = inhale_time + hold_time + exhale_time

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_phase)
        self.timer.start(50)  # 20 FPS

        self.start_time = time.time()
        self.current_radius = 0.2

    def advance_phase(self):
        t = (time.time() - self.start_time) % self.cycle_time
        if t < self.inhale_time:
            frac = t / self.inhale_time
            self.current_radius = 0.2 + 0.8 * frac
        elif t < self.inhale_time + self.hold_time:
            self.current_radius = 1.0
        else:
            t2 = t - (self.inhale_time + self.hold_time)
            frac = t2 / self.exhale_time
            self.current_radius = 1.0 - 0.8 * frac
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        size = min(w, h)
        radius = (size / 2.0) * self.current_radius

        color = QColor(100, 200, 255, 180)  # light blue
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)

        cx, cy = w / 2, h / 2
        painter.drawEllipse(
            int(cx - radius),
            int(cy - radius),
            int(2 * radius),
            int(2 * radius)
        )
        painter.end()


class ProgressiveColorBreathingCoach(QMainWindow):
    """
    Breathing coach that:
      - Calibrates peak threshold/min interval automatically
      - Detects peaks → accumulates peak timestamps
      - Computes IBIs, BPMs, HRV (SD of recent IBIs)
      - Colors a grayscale icon progressively from gray → green based on HRV
      - Guides breathing with an expanding/contracting circle
      - Shows raw PPG waveform in a pyqtgraph plot
    """
    def __init__(self,
                 port='/dev/cu.usbmodem1201',
                 window_samples=500,
                 fps=30,
                 calibrate_secs=60):
        super().__init__()
        self.setWindowTitle("Pulse + Guided Breathing for Kids")

        # 1) Arduino / PulseSensor setup
        self.board = Arduino(port)
        it = util.Iterator(self.board)
        it.start()
        self.analog0 = self.board.get_pin('a:0:i')

        # 2) Automatic calibration for first `calibrate_secs` seconds
        (self.PEAK_THRESHOLD,
         self.min_interval,
         self.BPM_LOW,
         self.BPM_HIGH,
         self.BPM_STD_THRESHOLD) = self.auto_calibrate(calibrate_secs)

        # 3) Prepare real-time data structures
        self.window_samples = window_samples
        self.raw_buffer = [0] * window_samples

        # Instead of a single last_peak_time, keep recent peak timestamps
        self.peak_times = deque(maxlen=50)  # store last 50 peak times (seconds)
        # Buffer of recent instantaneous BPMs for smoothing
        self.bpm_buffer = deque(maxlen=6)
        # <— Add this line to define how many BPMs to average:
        self.bpm_window = 6

        # 4) Load a single grayscale icon and attach a Colorize effect
        script_dir = Path(__file__).parent
        gray_icon_path = script_dir / "icon_gray.png"
        if gray_icon_path.exists():
            self.gray_pixmap = QPixmap(str(gray_icon_path))
        else:
            tmp = QPixmap(100, 100)
            tmp.fill(Qt.lightGray)
            self.gray_pixmap = tmp

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setPixmap(self.gray_pixmap)

        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.green)
        self.color_effect.setStrength(0.0)  # start fully gray
        self.status_label.setGraphicsEffect(self.color_effect)

        # 5) Build the GUI layout
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 5a) Raw waveform plot (PPG)
        self.plot_widget = pg.PlotWidget(title="PulseSensor A0 (Raw ADC)")
        self.curve = self.plot_widget.plot(self.raw_buffer, pen='r')
        self.plot_widget.setYRange(0, 1024)
        self.plot_widget.setLabel('left', "ADC Value")
        self.plot_widget.setLabel('bottom', "Sample Index")
        layout.addWidget(self.plot_widget, stretch=3)

        # 5b) Breathing guide widget (circle)
        self.breath_guide = BreathingGuideWidget(
            inhale_time=4.0, hold_time=1.0, exhale_time=6.0
        )
        self.breath_guide.setFixedHeight(200)
        layout.addWidget(self.breath_guide, stretch=2)

        # 5c) Grayscale icon with color-effect for stability
        layout.addWidget(self.status_label, stretch=1)

        # 5d) BPM text for debugging / tuning
        self.bpm_text = QLabel("BPM: --")
        self.bpm_text.setAlignment(Qt.AlignCenter)
        self.bpm_text.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.bpm_text, stretch=1)

        # 6) Timer for real-time updates (~30 Hz)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / fps))

    def auto_calibrate(self, secs: int):
        """
        Calibrate over `secs` seconds to determine:
          - PEAK_THRESHOLD (ADC value)
          - min_interval (seconds between valid peaks)
          - BPM_LOW, BPM_HIGH for a “calm” BPM range
          - BPM_STD_THRESHOLD for allowable BPM variability
        Steps:
            1. Sample raw ADC at ~30 Hz, store raw_values & timestamps
            2. Compute mean_raw + α·std_raw → peak_threshold (α starts at 1.0)
            3. Detect local-max peaks above that threshold
            4. If <3 peaks, increment α by 0.5 and retry (up to 4 times)
            5. From those peak timestamps, compute IBIs & BPM list
            6. avg_interval = mean(IBIs), avg_bpm, std_bpm
            7. min_interval = 0.8·avg_interval
               BPM_LOW = max(30, avg_bpm − 2·std_bpm)
               BPM_HIGH = avg_bpm + 2·std_bpm
               BPM_STD_THRESHOLD = 2·std_bpm
        """
        raw_values = []
        raw_times = []

        print(f"→ Calibrating for {secs} seconds (listening on A0)…")
        start_t = time.time()
        while time.time() - start_t < secs:
            v = self.analog0.read()
            if v is not None:
                raw = int(v * 1023)
                raw_values.append(raw)
                raw_times.append(time.time())
            time.sleep(1 / 30.0)

        if len(raw_values) < 5:
            raise RuntimeError("Calibration failed: not enough samples on A0.")

        arr = np.array(raw_values)
        mean_raw = np.mean(arr)
        std_raw = np.std(arr)
        alpha = 1.0
        peak_threshold = mean_raw + alpha * std_raw

        # Provisional peak detection in calibration buffer
        peak_times = []
        for i in range(1, len(arr) - 1):
            if arr[i] > peak_threshold and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peak_times.append(raw_times[i])

        attempts = 1
        while len(peak_times) < 3 and attempts < 4:
            alpha += 0.5
            peak_threshold = mean_raw + alpha * std_raw
            peak_times = []
            for i in range(1, len(arr) - 1):
                if arr[i] > peak_threshold and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                    peak_times.append(raw_times[i])
            attempts += 1

        if len(peak_times) < 2:
            # Fallback if too few peaks detected
            avg_interval = 1.0
            bpm_list = [60.0]
            std_bpm = 5.0
            avg_bpm = 60.0
        else:
            intervals = [
                peak_times[j] - peak_times[j - 1]
                for j in range(1, len(peak_times))
            ]
            intervals = [i for i in intervals if i > 0]
            avg_interval = float(np.mean(intervals))
            bpm_list = [60.0 / i for i in intervals]
            avg_bpm = float(np.mean(bpm_list))
            std_bpm = float(np.std(bpm_list))

        min_interval = 0.8 * avg_interval
        BPM_LOW = max(30.0, avg_bpm - 2.0 * std_bpm)
        BPM_HIGH = avg_bpm + 2.0 * std_bpm
        BPM_STD_THRESHOLD = 2.0 * std_bpm

        print(f"   → mean_raw={mean_raw:.1f}, std_raw={std_raw:.1f}")
        print(f"   → chosen peak_threshold={peak_threshold:.1f}")
        print(f"   → avg_interval={avg_interval:.3f}s → min_interval={min_interval:.3f}s")
        print(f"   → avg_bpm={avg_bpm:.1f}, std_bpm={std_bpm:.2f}")
        print(f"   → BPM_LOW={BPM_LOW:.1f}, BPM_HIGH={BPM_HIGH:.1f}, BPM_STD_THR={BPM_STD_THRESHOLD:.2f}")

        return peak_threshold, min_interval, BPM_LOW, BPM_HIGH, BPM_STD_THRESHOLD

    def update_frame(self):
        # 1) Read raw ADC from A0
        v = self.analog0.read()
        if v is not None:
            raw_adc = int(v * 1023)
        else:
            raw_adc = self.raw_buffer[-1]

        # 2) Update raw‐waveform buffer & redraw
        self.raw_buffer.append(raw_adc)
        if len(self.raw_buffer) > self.window_samples:
            self.raw_buffer.pop(0)
        self.curve.setData(self.raw_buffer)

        # 3) Peak detection & timestamp bookkeeping
        now = time.time()
        if (raw_adc > self.PEAK_THRESHOLD and
            (len(self.peak_times) == 0 or (now - self.peak_times[-1]) > self.min_interval)):

            # Record new peak time
            self.peak_times.append(now)

            # 4) If there is at least one previous peak, compute IBIs
            if len(self.peak_times) >= 2:
                ibi_list = [
                    self.peak_times[i] - self.peak_times[i - 1]
                    for i in range(1, len(self.peak_times))
                ]

                # 5) Convert IBIs → instantaneous BPMs
                bpm_list = [60.0 / ibi for ibi in ibi_list]

                # 6) Keep only the most recent BPM for smoothing
                recent_bpm = bpm_list[-1]
                self.bpm_buffer.append(recent_bpm)
                if len(self.bpm_buffer) > self.bpm_window:
                    self.bpm_buffer.popleft()

                # 7) Compute the smoothed (average) BPM
                avg_bpm = float(sum(self.bpm_buffer) / len(self.bpm_buffer))
                self.bpm_text.setText(f"BPM: {avg_bpm:.1f}")

                # 8) Compute HRV “stability” as SD of recent IBIs
                recent_ibis = ibi_list[-self.bpm_window:]
                std_ibi = float(np.std(recent_ibis))
                # Convert BPM_STD_THRESHOLD to an IBI‐std threshold (in seconds)
                max_allowed_ibi_std = (self.BPM_STD_THRESHOLD / 60.0)
                stability = (max_allowed_ibi_std - std_ibi) / max_allowed_ibi_std
                stability = max(0.0, min(1.0, stability))

                # 9) If average BPM is outside [BPM_LOW, BPM_HIGH], force stability = 0
                if not (self.BPM_LOW < avg_bpm < self.BPM_HIGH):
                    stability = 0.0

                # 10) Update the color-effect strength
                self.color_effect.setStrength(stability)

        # If no new peak, do nothing further this frame

    def closeEvent(self, event):
        self.board.exit()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = ProgressiveColorBreathingCoach(
        port='/dev/cu.usbmodem1201',  # adjust to your Arduino port
        window_samples=500,
        fps=30,
        calibrate_secs=60           # 60 s auto calibration
    )
    win.resize(800, 700)
    win.show()
    sys.exit(app.exec_())
