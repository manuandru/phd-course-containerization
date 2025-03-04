ARG PIXI_VERSION=0.41.4
FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION}

RUN mkdir -p /experiment
WORKDIR /experiment

COPY pyproject.toml pixi.lock /experiment/
RUN pixi install

COPY . /experiment

ENV DATA_DIR=/data
VOLUME $DATA_DIR
ENV OUTPUT_PATH=/output
VOLUME $OUTPUT_PATH
ENV OWNER=1000:1000

COPY --chmod=755 <<EOT /entrypoint.sh
#!/usr/bin/env bash
set -e
export OUTPUT_DIR=$OUTPUT_PATH/$(date +%Y-%m-%d)
mkdir -p \$OUTPUT_DIR
PYTHONUNBUFFERED=1 pixi run exec | tee \$OUTPUT_DIR/$(date +%H-%M-%S)-\$AGGREGATION_INTERVAL-ahead_\$AHEAD-\$AGGREGATION_METHOD.log
chown -R $OWNER \$OUTPUT_DIR
EOT
CMD ["/entrypoint.sh"]
