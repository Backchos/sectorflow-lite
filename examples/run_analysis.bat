@echo off
chcp 65001 >nul
echo ========================================
echo SectorFlow Lite - Stock Analysis Tool
echo ========================================
echo.
echo Starting analysis...
echo.
python simple_analysis.py
echo.
echo Analysis completed!
echo.
pause
