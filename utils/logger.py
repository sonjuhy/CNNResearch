import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs"):
    """
    터미널 출력과 파일 저장을 동시에 수행하는 로거(Logger)를 세팅합니다.
    매 학습(Run)마다 시간을 기준으로 새로운 로그 파일이 생성됩니다.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logger = logging.getLogger("CNNResearch")
    logger.setLevel(logging.INFO)
    
    # 이전에 설정된 핸들러가 있다면 제거 (중복 출력 방지)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 1. 터미널(Console) 출력 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 2. 텍스트 파일(File) 저장 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 로그 출력 포맷 설정 (시간 - 로그레벨 - 메시지)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"로거 세팅 완료. 텍스트 로그가 다음 경로에 자동 저장됩니다: {log_file}")
    
    return logger
