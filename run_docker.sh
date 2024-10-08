
# Build the Docker image
docker build -f Dockerfile.new --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg UNAME=$(whoami) -t vllm-perf .

UNAME=$(whoami)

docker run -v .:/opt/v \
           -v /mnt/nvdl/datasets/joc-datasets/unverified/model:/opt/model2 \
           -v /home/mlperf_inference_data/models/:/opt/model \
           -it --gpus=all --rm --ipc=host vllm-perf bash


RUN export HUGGING_FACE_HUB_TOKEN=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
# export HUGGING_FACE_HUB_TOKEN=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 && python benchmarks/benchmark_throughput.py --model=meta-llama/Llama-2-7b-hf --quantization=fp8 --input-len=20 --output-len=50  --enforce-eager