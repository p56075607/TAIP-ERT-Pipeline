cd C:\Users\Git\TAIP-ERT-Pipeline\scripts
call C:\Users\B30122\anaconda3\Scripts\activate.bat pg
call python run_pipeline_inversion.py --config ..\configs\site.yaml
pause