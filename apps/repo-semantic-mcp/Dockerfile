FROM python:3.12-slim

WORKDIR /repo
COPY apps/repo-semantic-mcp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /repo
ENV PYTHONPATH=/repo:/repo/libs
ENV PYTHONUNBUFFERED=1

CMD ["python", "apps/repo-semantic-mcp/main.py"]

