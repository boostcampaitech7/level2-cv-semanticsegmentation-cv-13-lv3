import threading
import time
import json
import subprocess
import requests
from pathlib import Path
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
import uvicorn

# .env 파일 로드
load_dotenv("server_ex.env")
print(f"Loaded Webhook URL: {os.getenv('WEBHOOK_GPU_URL')}")

class SingleServerGPUMonitor:
    def __init__(self):
        # 슬랙 Webhook URL과 서버 이름
        self.slack_webhook_url = os.getenv("WEBHOOK_GPU_URL")
        self.server_name = "Server4"  # 서버 이름
        
        # 데이터 경로 설정
        current_dir = Path(__file__).parent
        self.data_dir = current_dir / "gpu_monitor_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # 이전 값 파일 경로
        self.prev_values_file = self.data_dir / f"{self.server_name}_gpu_usage_prev.txt"
        self.initial_run_file = self.data_dir / f"{self.server_name}_gpu_initial_run.txt"

    def check_gpu_usage(self):
        """GPU 사용량 확인"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().replace(",", "/")
            else:
                print(f"Error running nvidia-smi: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("Timeout while running nvidia-smi")
            return None

    def send_slack_message(self, message):
        """슬랙 메시지 전송"""
        try:
            payload = {"text": message}
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10
            )
            if response.status_code != 200:
                print(f"Error sending Slack message: {response.status_code}")
        except Exception as e:
            print(f"Error sending Slack message: {e}")

    def monitor(self):
        """GPU 사용량 주기적 모니터링"""
        while True:
            current_value = self.check_gpu_usage()
            if current_value is None:
                print("Failed to retrieve GPU usage.")
                time.sleep(600)  # 10분 대기 후 재시도
                continue

            if not self.initial_run_file.exists():
                message = f"📊 {self.server_name}의 초기 GPU 상태:\n{current_value} MB"
                self.send_slack_message(message)
                with open(self.initial_run_file, 'w') as f:
                    f.write('done')
                with open(self.prev_values_file, 'w') as f:
                    json.dump({"usage": current_value}, f)
            else:
                try:
                    with open(self.prev_values_file, 'r') as f:
                        prev_value = json.load(f).get("usage", None)
                except (json.JSONDecodeError, FileNotFoundError):
                    prev_value = None

                if current_value != prev_value:
                    current_num = float(current_value.split('/')[0].strip())
                    prev_num = float(prev_value.split('/')[0].strip()) if prev_value else 0
                    change_in_gb = abs(current_num - prev_num) / 1024  # MB → GB
                    if change_in_gb >= 5:
                        change_type = "증가" if current_num > prev_num else "감소"
                        message = (
                            f"💻 GPU 사용량 변동 알림\n"
                            f"• 서버: {self.server_name}\n"
                            f"• 현재 사용량: {current_value} MB\n"
                            f"• 변동사항: {change_in_gb:.1f}GB {change_type}"
                        )
                        self.send_slack_message(message)

                with open(self.prev_values_file, 'w') as f:
                    json.dump({"usage": current_value}, f)

            time.sleep(600)  # 10분 대기

# FastAPI 설정
app = FastAPI()
monitor = SingleServerGPUMonitor()

# 기본 경로 추가
@app.get("/")
async def root():
    return {"message": "This is a GPU monitoring server. Use /gpu_status for GPU details."}

@app.post("/gpu_status")
async def gpu_status(request: Request):
    """실시간 GPU 상태 확인"""
    try:
        current_value = monitor.check_gpu_usage()
        if current_value is None:
            return {"text": "⚠️ GPU 상태를 가져오지 못했습니다. 다시 시도해주세요."}

        message = f"📊 {monitor.server_name}의 현재 GPU 상태:\n{current_value} MB"
        return {"text": message}
    except Exception as e:
        return {"text": f"⚠️ 오류 발생: {str(e)}"}

# 멀티스레드 실행
def run_monitor():
    """모니터링 스레드"""
    monitor.monitor()

def run_server():
    """FastAPI 서버 스레드"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # 각각의 기능을 스레드로 실행
    threading.Thread(target=run_monitor, daemon=True).start()
    threading.Thread(target=run_server, daemon=True).start()

    # 메인 스레드는 무기한 대기
    while True:
        time.sleep(1)
