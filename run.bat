@echo off
echo.
echo  =========================================
echo   ALPHA SCOUT — Small/Mid-Cap Screener
echo  =========================================
echo.
echo  Installing dependencies...
pip install -r requirements.txt --quiet
echo.
echo  Starting server at http://localhost:8000
echo  Press Ctrl+C to stop.
echo.
python main.py
