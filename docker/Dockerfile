FROM quay.io/jupyter/pytorch-notebook:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Asia/Tokyo

RUN conda install -y -c conda-forge pytorch-lightning

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root