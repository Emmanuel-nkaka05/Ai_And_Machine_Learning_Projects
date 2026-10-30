@echo off
REM 
REM 
setlocal
cd /d "%~dp0"

REM 
set PYCMD=py -3.12
%PYCMD% --version >nul 2>nul || set PYCMD=py -3.13
%PYCMD% --version >nul 2>nul || (
  echo Could not find Python 3.12 or 3.13 via the "py" launcher.
  echo Install Python 3.12 from python.org, tick "Add python.exe to PATH", then run this again.
  pause & exit /b 1
)

echo [1/4] Creating build environment...
%PYCMD% -m venv build_venv || (pause & exit /b 1)
call build_venv\Scripts\activate.bat

echo [2/4] Installing packages (needs internet)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller || (pause & exit /b 1)

echo [3/4] Building the application...
pyinstaller --noconfirm --clean --windowed --name OpticNerve ^
  --add-data "models;models" --add-data "samples;samples" ^
  recognize_gui.py || (pause & exit /b 1)

echo [4/4] Bundling Tesseract (optional, makes the app fully portable)...
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
  xcopy /E /I /Y /Q "C:\Program Files\Tesseract-OCR" "dist\OpticNerve\Tesseract-OCR" >nul
  echo     Tesseract copied.
) else (
  echo     Tesseract not found - install it first if you want text recognition inside the app folder.
)

echo.
echo DONE.  Run:  dist\OpticNerve\OpticNerve.exe
echo Share the whole dist\OpticNerve folder (zip it).
pause
