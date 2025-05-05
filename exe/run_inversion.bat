cd C:\TAIP-ERT-Pipeline\scripts
call C:\Users\TAIP_DataCenter\anaconda3\Scripts\activate.bat pg
call python run_pipeline_inversion.py --config ..\configs\site.yaml
pause