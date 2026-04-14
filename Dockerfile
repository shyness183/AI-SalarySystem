FROM python:3.11-slim

WORKDIR /app

# 用国内镜像源解决 apt-get 网络问题
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web/ /app/

EXPOSE 5000
CMD ["python", "app.py"]