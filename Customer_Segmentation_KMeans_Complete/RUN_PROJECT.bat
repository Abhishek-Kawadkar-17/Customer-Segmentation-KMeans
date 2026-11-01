@echo off
cd /d "%~dp0"
echo Installing required Python libraries...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
echo Installation failed. Check Python and internet connection.
  pause
  exit /b 1
)
echo.
echo Running Customer Segmentation project...
python customer_segmentation.py
echo.
echo Finished. Check the outputs and results folders.
pause
