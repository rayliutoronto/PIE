#!/bin/bash

cd /PIE/PIE
tensorflow_model_server  --port=9000 --model_name=pie --model_base_path=/PIE/output/export/BestExport/ &> ../output/serving.log &
gunicorn --workers=20 api:app -b localhost:19000 &> ../output/web_server.log &
