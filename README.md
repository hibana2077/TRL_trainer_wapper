<!--
 * @Author: hibana2077 hibana2077@gmail.com
 * @Date: 2024-04-20 10:54:54
 * @LastEditors: hibana2077 hibana2077@gmail.com
 * @LastEditTime: 2024-04-20 11:19:20
 * @FilePath: \TRL_trainer_wapper\README.md
 * @Description: 
-->
# TRL trainer wapper

This is a simple tool for using TRL to training a Large Language Model (LLM) on cloud GPUs.

## Prerequisites

- Docker
    - Nvidia docker runtime
    - Image
        - pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
        - nvidia/cuda:12.1.0-runtime-ubuntu22.04

## Quick Start

Create a new container with the following command:

```bash
docker run --runtime=nvidia --name trl-trainer -dt pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
```

Then, enter the container:

```bash
docker exec -it trl-trainer bash
```

Clone this repository:

```bash
git clone https://github.com/hibana2077/TRL_trainer_wapper.git
```

cd into the repository and run setup:

```bash
cd TRL_trainer_wapper
bash ./setup.sh
```

Finally, run the training script:

```bash
cd src
python3 ./sft_trainer.py
```

## Documentation

[TRL_trainer_wapper](https://trl-trainer-wapper.hibana2077.com)

## License

[MIT](https://opensource.org/licenses/MIT)