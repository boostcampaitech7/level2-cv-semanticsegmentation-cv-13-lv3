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

load_dotenv("server_ex.env")
print(f"Loaded Webhook URL: {os.getenv('WEBHOOK_GPU_URL')}")