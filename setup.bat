@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Done. Now:  ollama pull qwen2.5:3b-instruct
echo Then:         run.bat
endlocal