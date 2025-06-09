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
    QGraphicsOpacityEffect,
    QGridLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QColor, QPainter, QImage

import numpy as np
import pyqtgraph as pg
from pyfirmata import Arduino, util


class BreathingGuideWidget(QWidget):
    """
    A circle that expands (inhale), holds, then shrinks (exhale).
    Smaller range for a more compact display.
    """
    def __init__(self, inhale_time=4.0, hold_time=1.0, exhale_time=6.0, parent=None):
        super().__init__(parent)
        self.inhale_time = inhale_time
        self.hold_time = hold_time
        self.exhale_time = exhale_time
        self.cycle_time = inhale_time + hold_time + exhale_time

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_phase)
        self.timer.start(50)  # 20 FPS

        self.start_time = time.time()
        self.current_radius = 0.3  # base radius

    def advance_phase(self):
        t = (time.time() - self.start_time) % self.cycle_time
        if t < self.inhale_time:
            frac = t / self.inhale_time
            self.current_radius = 0.3 + 0.4 * frac
        elif t < self.inhale_time + self.hold_time:
            self.current_radius = 0.7
        else:
            frac = (t - self.inhale_time - self.hold_time) / self.exhale_time
            self.current_radius = 0.7 - 0.4 * frac
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        r = min(w, h) * self.current_radius / 2
        painter.setBrush(QColor(100, 200, 255, 180))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(w/2 - r), int(h/2 - r), int(2*r), int(2*r))
        painter.end()


class ProgressiveColorBreathingCoach(QMainWindow):
    def __init__(
        self,
        port='/dev/cu.usbmodem1201',
        window_samples=500,
        fps=30,
        calibrate_secs=60
    ):
        super().__init__()
        self.setWindowTitle("Pulse + Guided Breathing for Kids")

        # Arduino / PulseSensor setup
        self.board = Arduino(port)
        it = util.Iterator(self.board)
        it.start()
        self.analog0 = self.board.get_pin('a:0:i')

        # Automatic calibration
        (
            self.PEAK_THRESHOLD,
            self.min_interval,
            self.BPM_LOW,
            self.BPM_HIGH,
            self.BPM_STD_THRESHOLD
        ) = self.auto_calibrate(calibrate_secs)

        # Data buffers
        self.window_samples = window_samples
        self.raw_buffer = [0] * window_samples
        self.peak_times = deque(maxlen=50)
        self.bpm_buffer = deque(maxlen=6)
        self.bpm_window = 6

        # Load and scale nature image + grayscale copy
        script_dir = Path(__file__).parent
        img_path = script_dir / 'nature.jpg'
        orig = QPixmap(str(img_path))
        # scale to target width
        target_w = 512
        color = orig.scaledToWidth(target_w, Qt.SmoothTransformation)
        gray_img = orig.toImage().convertToFormat(QImage.Format_Grayscale8)
        gray = QPixmap.fromImage(gray_img).scaledToWidth(target_w, Qt.SmoothTransformation)

        # stacked labels for grayscale + color overlay
        self.img_container = QWidget()
        self.img_container.setFixedSize(color.size())
        grid = QGridLayout(self.img_container)
        grid.setContentsMargins(0, 0, 0, 0)
        self.gray_label = QLabel(); self.gray_label.setPixmap(gray)
        grid.addWidget(self.gray_label, 0, 0)
        self.color_label = QLabel(); self.color_label.setPixmap(color)
        self.opacity = QGraphicsOpacityEffect(); self.opacity.setOpacity(0.0)
        self.color_label.setGraphicsEffect(self.opacity)
        grid.addWidget(self.color_label, 0, 0)

        # Build GUI
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        pg.setConfigOption('background', '#222'); pg.setConfigOption('foreground', 'w')

        # Raw PPG
        self.plot = pg.PlotWidget(title="PulseSensor A0 (Raw ADC)")
        self.plot.setYRange(0, 1024)
        self.plot.setLabel('left', 'ADC Value'); self.plot.setLabel('bottom', 'Sample Index')
        self.curve = self.plot.plot(self.raw_buffer, pen='r')
        layout.addWidget(self.plot, stretch=3)

        # Breathing circle centered
        self.breath = BreathingGuideWidget(3, 1, 3)
        self.breath.setFixedSize(150, 150)
        layout.addWidget(self.breath, alignment=Qt.AlignHCenter, stretch=2)

        # Image centered under circle
        layout.addWidget(self.img_container, alignment=Qt.AlignHCenter, stretch=1)

        # BPM text
        self.bpm_text = QLabel("BPM: --"); self.bpm_text.setAlignment(Qt.AlignCenter)
        self.bpm_text.setStyleSheet("font-size:16px;color:white;")
        layout.addWidget(self.bpm_text, stretch=1)

        # Timer
        t = QTimer(self); t.timeout.connect(self.update_frame); t.start(int(1000 / fps))

    def auto_calibrate(self, secs):
        raw, times = [], []
        t0 = time.time()
        while time.time() - t0 < secs:
            v = self.analog0.read()
            if v is not None:
                raw.append(int(v * 1023)); times.append(time.time())
            time.sleep(1/30)
        arr = np.array(raw); mu, sd = arr.mean(), arr.std()
        thr = mu + sd
        def det(t):
            pts = []
            for i in range(1, len(arr) - 1):
                if arr[i] > t and arr[i] > arr[i-1] and arr[i] > arr[i+1]: pts.append(times[i])
            return pts
        pts, alpha = det(thr), 1.0
        while len(pts) < 3 and alpha < 3.0:
            alpha += 0.5; thr = mu + alpha * sd; pts = det(thr)
        if len(pts) < 2: return thr, 1.0, 30.0, 90.0, 5.0
        ibis = np.diff(pts); ibis = ibis[ibis>0]
        avg_i = ibis.mean(); bpms = 60/ibis; avg_b, std_b = bpms.mean(), bpms.std()
        return thr, 0.8*avg_i, max(30, avg_b-2*std_b), avg_b+2*std_b, 2*std_b

    def update_frame(self):
        v = self.analog0.read(); r = int(v*1023) if v else self.raw_buffer[-1]
        self.raw_buffer.append(r)
        if len(self.raw_buffer) > self.window_samples: self.raw_buffer.pop(0)
        self.curve.setData(self.raw_buffer)
        now = time.time()
        if r > self.PEAK_THRESHOLD and (not self.peak_times or now - self.peak_times[-1] > self.min_interval):
            self.peak_times.append(now)
            if len(self.peak_times) >= 2:
                ibis = np.diff(list(self.peak_times)); bpm = 60/ibis[-1]
                self.bpm_buffer.append(bpm)
                avg_bpm = sum(self.bpm_buffer)/len(self.bpm_buffer)
                self.bpm_text.setText(f"BPM: {avg_bpm:.1f}")
                recent = ibis[-self.bpm_window:]; std_i = np.std(recent)
                stability = (self.BPM_STD_THRESHOLD/60 - std_i)/(self.BPM_STD_THRESHOLD/60)
                stability = max(0.0, min(1.0, stability))
                if not(self.BPM_LOW < avg_bpm < self.BPM_HIGH): stability=0.0
                self.opacity.setOpacity(stability)

    def closeEvent(self, event):
        self.board.exit(); super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ProgressiveColorBreathingCoach(
        port='/dev/cu.usbmodem1201', window_samples=500, fps=30, calibrate_secs=60
    )
    w.resize(1400,1000); w.show(); sys.exit(app.exec_())