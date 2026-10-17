@echo off
if not exist .venv\Scripts\python.exe (
  echo No virtualenv found - run setup.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
python main.py %*