@echo off
chcp 65001 >nul
echo ========================================
echo SectorFlow Lite - Web Server
echo ========================================
echo.
echo Starting web server...
echo.
echo After server starts, open browser and go to:
echo http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.
python web_analysis.py
pause

