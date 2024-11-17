from multiprocessing import Process
from monitor_gpu import monitor_gpu
from monitor_disk import monitor_disk

if __name__ == "__main__":
    # GPU 모니터링 프로세스
    gpu_process = Process(target=monitor_gpu)
    
    # 디스크 모니터링 프로세스
    disk_process = Process(target=monitor_disk)
    
    # 프로세스 시작
    gpu_process.start()
    disk_process.start()
    
    # 프로세스가 종료될 때까지 대기
    gpu_process.join()
    disk_process.join()
