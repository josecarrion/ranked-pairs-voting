@echo off
REM Build script for Ranked Pairs Voting App
REM Run this on Windows to create the standalone executable

echo Installing build dependencies...
pip install pyinstaller flet pandas networkx openpyxl graphviz

echo.
echo Building executable with flet pack...
flet pack voting_app.py --name "RankedPairsVoting" --add-data "graphviz_bin;graphviz_bin" --add-data "ranked_pairs_voting.py;." -y

echo.
if exist "dist\RankedPairsVoting.exe" (
    echo Build complete!
    echo The executable is in the 'dist' folder: dist\RankedPairsVoting.exe
) else (
    echo Build may have failed. Check the output above for errors.
)
echo.
pause
