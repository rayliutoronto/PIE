#!/bin/bash

cd /PIE/PIE
tensorflow_model_server  --port=9000 --model_name=pie --model_base_path=/PIE/output/export/BestExport/ &> ../output/serving.log &
gunicorn --workers=9 api:app -b 0.0.0.0:19999 &> ../output/web_server.log &
