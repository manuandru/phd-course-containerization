# Containerization PhD course delivery

This repository contains the exercise for **Containerisation and Orchestration for Research Reproducibility** course in the PhD program in Computer Science and Engineering at the University of Bologna.

## Tools

- [Docker](https://www.docker.com)
- [Pixi](https://pixi.sh/latest/): for python environment management

## Experiment

The experiment consists of aggregating by datetime the dataset in different ways and then many models are tested on the aggregated dataset.

> For the sake of simplicity, the dataset is pushed into the repository.

## Execution

To execute the experiment, run the following command:

```bash
docker compose up
```

The experiment has been tested on the following machines:

- MacOS Sequoia 15.3.1 Apple M1 Pro
- Ubuntu 22.04 amd64
