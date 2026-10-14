FROM python:3.10-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY run.py run.sh model_inference.py featurize.py test_ids.json ./
COPY model/ ./model/
ENTRYPOINT ["bash", "/app/run.sh"]
CMD ["--out-dir", "/output/reproduced", "--cache-dir", "/output/cache"]
