import os
import subprocess
import requests
import time
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv('server_ex.env')

# 환경 변수 읽기
WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SERVER_NAME = os.getenv("SERVER_NAME", "Unknown Server")

if not WEBHOOK_URL:
    raise ValueError("SLACK_WEBHOOK_URL 환경 변수가 설정되지 않았습니다.")

# 설정 값
CHECK_INTERVAL = 300  # 5분 간격
THRESHOLD_PERCENT = 90  # 디스크 사용량 임계치 (%)
INCREASE_THRESHOLD = 10  # 이전 상태 대비 % 증가 임계치
TARGET_MOUNT = "/data/ephemeral"  # 모니터링할 대상 경로

last_usage = None  # 이전 디스크 상태 기록

# 데이터 저장 경로 설정
LOG_DIR = "monitor_log"
LOG_FILE = os.path.join(LOG_DIR, "disk_status.log")
os.makedirs(LOG_DIR, exist_ok=True)

def get_disk_usage(target_mount):
    """특정 마운트 경로의 디스크 사용량 조회"""
    try:
        result = subprocess.run(
            ["df", "-h", target_mount],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return None  # 유효한 결과 없음

        # 데이터 파싱
        _, size, used, avail, percent, mount = lines[1].split()
        percent_value = int(percent.strip('%'))  # '13%' -> 13
        return {"size": size, "used": used, "avail": avail, "percent": percent_value, "mount": mount}
    except Exception as e:
        print(f"디스크 사용량 조회 중 오류 발생: {e}")
        return None

def save_to_file(data):
    """디스크 상태를 파일에 저장"""
    with open(LOG_FILE, "a") as f:
        f.write(data + "\n")

def send_to_slack(message):
    """슬랙으로 메시지 전송"""
    full_message = f"🖥️ *{SERVER_NAME}*\n{message}"
    requests.post(WEBHOOK_URL, json={"text": full_message})

def monitor_disk():
    global last_usage
    while True:
        current_usage = get_disk_usage(TARGET_MOUNT)
        if not current_usage:
            print(f"{TARGET_MOUNT} 경로를 가져올 수 없습니다.")
            time.sleep(CHECK_INTERVAL)
            continue

        # 알림 메시지 생성
        message = (
            f"📊 *{current_usage['mount']}* 디스크 상태:\n"
            f"- 전체 용량: {current_usage['size']}\n"
            f"- 사용 중: {current_usage['used']}\n"
            f"- 가용 용량: {current_usage['avail']}\n"
            f"- 점유율: {current_usage['percent']}%"
        )

        alerts = []
        if current_usage["percent"] >= THRESHOLD_PERCENT:
            alerts.append("⚠️ 90% 초과! 디스크 점유율이 높습니다. ⚠️")

        # 이전 사용량 대비 증가 여부 확인
        if last_usage:
            prev_percent = last_usage["percent"]
            increase = current_usage["percent"] - prev_percent
            if increase > 0:
                alerts.append(f"🔼 이전보다 {increase}% 증가했습니다. 🔼")

        # 최종 메시지 생성 및 전송
        if alerts:
            message += "\n" + "\n".join(alerts)
        send_to_slack(message)
        save_to_file(message)

        # 상태 갱신
        last_usage = current_usage

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    monitor_disk()
