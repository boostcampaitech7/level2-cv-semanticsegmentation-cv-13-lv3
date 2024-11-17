import subprocess
import time
import json
import requests
from pathlib import Path
import os

from dotenv import load_dotenv

# server_ex.env 파일을 지정하여 로드
load_dotenv("server_ex.env")

class GPUMonitor:
    def __init__(self):
        # Slack Webhook URL
        self.slack_webhook_url = os.getenv("WEBHOOK_GPU_URL")
        
        # 서버 정보 딕셔너리
        # ssh -i [키파일 경로] -p [포트넘버] root@[서버주소]
        self.servers = {
            "Server1": [os.getenv("SERVER_KEY_PATH"), os.getenv("SERVER1_PORT"), os.getenv("SERVER1_ADDRESS")],
            "Server2": [os.getenv("SERVER_KEY_PATH"), os.getenv("SERVER2_PORT"), os.getenv("SERVER2_ADDRESS")],
            "Server3": [os.getenv("SERVER_KEY_PATH"), os.getenv("SERVER3_PORT"), os.getenv("SERVER3_ADDRESS")],
            "Server4": [os.getenv("SERVER_KEY_PATH"), os.getenv("SERVER4_PORT"), os.getenv("SERVER4_ADDRESS")]
        }

        # 현재 파일의 디렉토리를 기준으로 data_dir 설정
        current_dir = Path(__file__).parent
        self.data_dir = current_dir / "gpu_monitor_data"
        self.data_dir.mkdir(exist_ok=True)  # 디렉토리가 없으면 생성
        
        # 이전 값과 초기 실행 여부 파일 경로 설정
        self.prev_values_file = self.data_dir / "gpu_usage_prev.txt"
        self.current_values_file = self.data_dir / "gpu_usage_current.txt"
        
        # 초기 실행 여부를 확인하기 위한 플래그 파일
        self.initial_run_file = self.data_dir / "gpu_initial_run.txt"

    def check_gpu_usage(self):
        """각 서버의 디스크 사용량을 확인하고 저장"""
        current_values = {}

        for server_name, server_info in self.servers.items():
            key_path, port, ip = server_info
            
            try:
                # SSH 명령어 구성 (StrictHostKeyChecking=no 추가)
                ssh_command = [
                    "ssh",
                    "-i", key_path,
                    "-p", port,
                    "-o", "StrictHostKeyChecking=no",  # 호스트 키 검증 건너뛰기
                    "-o", "UserKnownHostsFile=/dev/null",  # known_hosts 파일 무시
                    "-o", "ServerAliveInterval=60",  # 60초마다 연결 상태 확인
                    "-o", "ServerAliveCountMax=3",   # 최대 3회까지 상태 확인 실패 허용
                    f"root@{ip}",
                    "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
                ]
                
                # 명령어 실행
                result = subprocess.run(
                    ssh_command,
                    capture_output=True,
                    text=True,
                    timeout=30 # 타임아웃 설정
                )
                
                if result.returncode == 0:
                    #usage = result.stdout.strip()
                    usage = result.stdout.strip().replace(",", "/")  # 콤마를 슬래시로 변경
                    if usage:
                        current_values[server_name] = usage
                else:
                    print(f"Error on {server_name}: {result.stderr}")
            
            except subprocess.TimeoutExpired:
                print(f"Timeout while connecting to {server_name}")
            except Exception as e:
                print(f"Error checking {server_name}: {e}")
        
        return current_values

    def send_slack_message(self, message):
        """Slack으로 메시지 전송"""
        try:
            payload = {"text": message}
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10 # 타임아웃 설정
            )
            if response.status_code != 200:
                print(f"Error sending Slack message: {response.status_code}")
        except Exception as e:
            print(f"Error sending Slack message: {e}")

    def send_initial_status(self, current_values):
        """모든 서버의 초기 상태를 Slack으로 전송"""
        message = "📊 현재 모든 서버의 GPU 사용량 현황\n\n"
        for server_name, value in current_values.items():
            message += f"• {server_name}: {value} MB\n"
        self.send_slack_message(message)

    def run(self):
        """메인 모니터링 로직"""
        
        # 현재 값 확인
        current_values = self.check_gpu_usage()

        # 초기 실행 체크
        if not self.initial_run_file.exists():
            # 초기 상태 전송
            self.send_initial_status(current_values)
            # 초기 실행 표시
            with open(self.initial_run_file, 'w') as f:
                f.write('done')
            # 현재 값을 이전 값으로 저장
            with open(self.prev_values_file, 'w') as f:
                json.dump(current_values, f)
            return

        try:
            # 이전 값 로드
            with open(self.prev_values_file, 'r') as f:
                prev_values = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading previous values: {e}")
            prev_values = {}
            
        # 변경사항 확인 및 알림
        for server_name, current_value in current_values.items():
            prev_value = prev_values.get(server_name)
            
            if current_value != prev_value:
                try:
                    # 슬래시로 구분된 두 값을 나누어 사용량과 총 용량을 각각 추출
                    current_num = float(current_value.split('/')[0].strip())  # 현재 사용량 (MB)
                    prev_num = float(prev_value.split('/')[0].strip()) if prev_value else 0
                    
                    # 변화량 계산
                    change = current_num - prev_num
                    change_in_gb = abs(change) / 1024  # MB에서 GB로 변환
                    
                    # 5GB 이상 변화가 있는 경우에만 Slack 메시지 발송
                    if change_in_gb >= 5:
                        change_str = f"({'증가' if change > 0 else '감소'}: {change_in_gb:.1f}GB)"
                        message = (
                            f"💻 GPU 사용량 변동 알림\n"
                            f"• 서버: {server_name}\n"
                            f"• 현재 사용량: {current_value} MB\n"
                            f"• 이전 사용량: {prev_value if prev_value else '정보 없음'} MB\n"
                            f"• 변동사항: {change_str}"
                        )
                        self.send_slack_message(message)
                
                except (ValueError, AttributeError):
                    print(f"Error processing values for {server_name}")
                    
        # 현재 값을 이전 값으로 저장
        with open(self.prev_values_file, 'w') as f:
            json.dump(current_values, f)

def main():
    monitor = GPUMonitor()
    
    while True:
        try:
            monitor.run()
            print(f"모니터링 실행 완료: {time.strftime('%Y-%m-%d %H:%M:%S')}")  # 타임스탬프 추가
            time.sleep(600)  # 10분 대기
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(600)  # 에러 발생시에도 10분 대기 후 재시도

if __name__ == "__main__":
    main()