# Code only. Data is mounted at run time and never copied into the image.
FROM public.ecr.aws/docker/library/python:3.12.7-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY organism ./organism
RUN pip install --no-cache-dir numpy==2.1.3 && pip install --no-cache-dir --no-deps .
WORKDIR /work
ENTRYPOINT ["organism"]
CMD ["--help"]
