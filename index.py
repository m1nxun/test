#!/usr/bin/env python3
# =====================================================================
#  실시간 청각 보조 - 라즈베리파이
#    USB 마이크 입력 -> FFT 로 음역 분해 -> 특정 음역 계수 조절
#    -> 역FFT -> 오버랩-애드로 이어붙임 -> Aux(3.5mm) 출력
#
#  준비(세팅법은 파일 아래 주석과 함께 제공된 안내문 참고):
#    sudo apt install -y libportaudio2 python3-numpy
#    pip3 install sounddevice numpy
#
#  실행:  python3 realtime_hearing_assist.py
#  중지:  Ctrl + C
#
#  주의: 마이크와 스피커가 가까우면 삑 소리(하울링)가 난다.
#        반드시 이어폰/헤드폰으로 들을 것.
# =====================================================================

import numpy as np

# ====== 설정값 (필요에 맞게 바꾸기) ======
SR        = 44100      # 표본화 주파수
N         = 1024       # FFT 크기(한 토막 길이). 클수록 음질↑ 지연↑
HOP       = N // 2     # 토막을 절반씩 겹침(50% 오버랩-애드)

LOW_CUT   = 120        # 이 주파수(Hz) 아래는 걸러냄(웅웅거리는 저주파 소음)
BOOST_FREQ= 2000       # 이 주파수(Hz) 이상을 키움(잘 안 들리는 높은 음)
BOOST_GAIN= 3.0        # 높은 음을 몇 배로 키울지
MASTER_VOL= 0.6        # 전체 음량(0.0~1.0). 청력 보호를 위해 너무 키우지 말 것

# 입출력 장치 지정(이름 일부). None 이면 시스템 기본 장치 사용.
# query_devices() 결과(아래 안내 참고)를 보고 정확한 이름 일부를 넣으면 됨.
INPUT_DEVICE  = None   # 예: "USB"  (USB 마이크 이름에 보통 USB 포함)
OUTPUT_DEVICE = None   # 예: "Headphones" 또는 "bcm2835"  (3.5mm Aux)


# ====== 주파수별 이득 곡선 만들기 ======
# 부드러운 경사로 연결해 갑작스러운 잡음(아티팩트)을 줄인다.
def build_gain_curve(sr, n):
    freqs = np.fft.rfftfreq(n, 1 / sr)
    # 제어점: (주파수, 이득). 사이는 직선으로 부드럽게 잇는다.
    control_f = [0,   LOW_CUT*0.7, LOW_CUT, BOOST_FREQ*0.8, BOOST_FREQ, sr/2]
    control_g = [0.0, 0.0,        1.0,     1.0,            BOOST_GAIN,  BOOST_GAIN]
    return np.interp(freqs, control_f, control_g)


# ====== 실시간 신호 처리기 (오버랩-애드) ======
class Processor:
    def __init__(self, sr, n, hop):
        self.n = n
        self.hop = hop
        # 주기형 한(Hann) 창: 50% 겹침에서 창들의 합이 일정해진다
        self.window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
        # 겹쳐 더했을 때 생기는 일정한 배수(COLA)를 구해 나중에 나눠 보정
        norm = np.zeros(n)
        for k in range(n // hop):
            norm += np.roll(self.window, k * hop)
        self.cola = norm.mean()              # 주기형 한 창·50% 겹침이면 ≈ 1.0
        self.in_buf = np.zeros(n)            # 최근 n개 입력 표본
        self.ola = np.zeros(n)               # 출력 누적(오버랩-애드) 버퍼
        self.gain = build_gain_curve(sr, n)

    def process(self, x):
        # x: 새로 들어온 hop개의 표본(모노)
        # 1) 입력 버퍼를 hop 만큼 밀고 새 표본을 끝에 채움
        self.in_buf[:-self.hop] = self.in_buf[self.hop:]
        self.in_buf[-self.hop:] = x
        # 2) 창을 곱하고 FFT 로 음역 분해
        frame = self.in_buf * self.window
        spec = np.fft.rfft(frame)
        # 3) 음역별 계수 조절(핵심: 특정 음역만 키우거나 줄임)
        spec *= self.gain
        # 4) 역FFT 로 다시 소리로 (겹침 배수 보정)
        rec = np.fft.irfft(spec, n=self.n) / self.cola
        # 5) 오버랩-애드: 누적 버퍼에 더하고 앞쪽 hop개를 출력
        self.ola += rec
        out = self.ola[:self.hop].copy()
        self.ola[:-self.hop] = self.ola[self.hop:]
        self.ola[-self.hop:] = 0.0
        return out


def main():
    import sounddevice as sd

    proc = Processor(SR, N, HOP)

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status)
        mono = indata[:, 0]                       # 마이크(모노) 입력
        out = proc.process(mono) * MASTER_VOL     # 처리 후 음량 적용
        np.clip(out, -0.98, 0.98, out=out)        # 출력 상한(청력/스피커 보호)
        outdata[:, 0] = out

    print("실시간 청각 보조 시작. 멈추려면 Ctrl+C.")
    print(f"설정: {LOW_CUT}Hz 아래 차단, {BOOST_FREQ}Hz 이상 {BOOST_GAIN}배 증폭")
    # 모노로 입출력. 라즈베리파이 ALSA 가 모노를 양쪽 이어폰으로 자동 전달함.
    with sd.Stream(samplerate=SR, blocksize=HOP, dtype="float32",
                   channels=1,
                   device=(INPUT_DEVICE, OUTPUT_DEVICE),
                   callback=callback):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\n종료합니다.")


if __name__ == "__main__":
    main()
