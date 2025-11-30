@echo off
echo Running CRF Test...
python main.py --model crf --mode train --max_iter 5
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Running BiLSTM Test (1 epoch)...
python main.py --model bilstm --mode train --epochs 1 --batch_size 16
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo All tests passed!
