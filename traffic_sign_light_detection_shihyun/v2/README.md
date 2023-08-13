# Ultralytics YOLOv5 traffic sign and light detection

1. Download dataset by reading 데이터_다운로드.pdf
2. Resize images to half of original size (1280x720 --> 640x360)
3. Change directory structure according to `process_data.py` and `dataset.yaml`
4. Copy `datasets` as `unprocessed_datasets`
5. (Optional) remove images from `unprocessed_datasets`
6. Run `process_data.py` to preprocess data
7.  ```bash
    $ git clone https://github.com/ultralytics/yolov5.git
    $ cd yolov5
    $ cp ../dataset.yaml .
    $ cd ..
    $ tar -cf upload.tar datasets yolov5
    ```
8. Upload `upload.tar` to Google Drive
9. Run `traffic_lights_signs_detection.ipynb` in Google Colab
10. Extract downloaded `runs.tar` and move `runs` to same directory as `traffic_lights_signs.py`
11. Run `traffic_lights_signs.py`
