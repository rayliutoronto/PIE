#!/bin/bash

cd /PIE/PIE
nohup tensorflow_model_server  --port=9000 --model_name=pie --model_base_path=/PIE/output/export/BestExport/ > serving.log &
nohup gunicorn --workers=20 api:app -b localhost:19000 > web_server.log &
