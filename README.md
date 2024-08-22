# Team hierocles-of-alexandria for ValueEval'24

## Dockerization of Prediction Model
- uses model from [https://huggingface.co/SotirisLegkas/multi-head-xlm-xl-tokens-38](https://huggingface.co/SotirisLegkas/multi-head-xlm-xl-tokens-38)

```bash
# run
tira-run \
  --input-directory "$PWD/valueeval24/test" \
  --output-directory "$PWD/output" \
  --image webis/valueeval24-hierocles-of-alexandria:1.0.0

# or
docker run --rm \
  -v "$PWD/valueeval24/test:/dataset" -v "$PWD/output:/output" \
  webis/valueeval24-hierocles-of-alexandria:1.0.0

# view results
cat output/run.tsv
```

## Inference Server
A local inference server can be started from the same Docker-Image using:
```bash
PORT=8001

docker run --rm -it --init \
  -v "$PWD/logs:/logs" \
  -p $PORT:$PORT \
  --entrypoint tira-run-inference-server \
  webis/valueeval24-hierocles-of-alexandria:1.0.0 \
  --script /predict.py --port $PORT

# or, for zero-shot version
docker run --rm -it --init \
  -v "$PWD/logs:/logs" \
  -p $PORT:$PORT \
  -e HOA_ZERO_SHOT="True" \
  --entrypoint tira-run-inference-server \
  webis/valueeval24-hierocles-of-alexandria:1.0.0 \
  --script /predict.py --port $PORT
docker run --rm -it --init -v "$PWD/logs:/logs" -p 8001:8001 -e HOA_ZERO_SHOT="True" --entrypoint tira-run-inference-server valueeval24-hierocles-of-alexandria:1.0.0 --script /predict.py --port 8001
```

Exemplary request for a server running on localhost:8001 are

```bash
# POST (JSON list as payload)
curl -X POST -H "application/json" \
  -d "[{\"Text\": \"element 1\", \"language\": \"EN\"}, {\"Text\": \"element 2\", \"language\": \"EN\"}]" \
  localhost:8001
```
and
```bash
# GET (JSON object string(s) passed to the 'payload' parameter)
curl "localhost:8001?payload=\"element+1\"&payload=\"element+2\""
```
The possible values for language are `EN`, `EL`, `DE`, `TR`, `FR`, `BG`, `HE`, `IT`, `NL`.
Please note that GET-request are currently only possible for language `EN`.


## Building
Both `webis/valueeval24-hierocles-of-alexandria:1.0.0` and `webis/valueeval24-hierocles-of-alexandria:1.0.0-no-model` are available on dockerhub.

The `webis/valueeval24-hierocles-of-alexandria:1.0.0-no-model` image is much smaller, but requires you to:
- use `download_model_and_tokenizer.py` separately
- mount the created `models` and `tokenizer` directories into the containers root directory

```bash
# build stand-alone image
docker build -f Dockerfile -t webis/valueeval24-hierocles-of-alexandria:1.0.0 .

# build image without models
DOCKER_BUILDKIT=1 docker build -f Dockerfile \
  -t webis/valueeval24-hierocles-of-alexandria:1.0.0-no-model \
  --build-arg include_models=false .
```



## Internal
Running the image in our Webis cluster on a GPU.
```
srun --container-writable --mem=32g --cpus-per-task=8 --gres=gpu:3g.20gb:1 --container-image=webis/valueeval24-hierocles-of-alexandria:1.0.0-no-model \
  /bin/bash -c "ln -s /mnt/ceph/storage/data-in-production/data-research/computational-ethics/valueeval/hierocles-of-alexandria24/models/SotirisLegkas /models/SotirisLegkas;ln -s /mnt/ceph/storage/data-in-production/data-research/computational-ethics/valueeval/hierocles-of-alexandria24/tokenizer/SotirisLegkas /tokenizer/SotirisLegkas;ln -s /mnt/ceph/storage/data-in-production/data-research/computational-ethics/valueeval/hierocles-of-alexandria24/logs;WORLD_SIZE=1 RANK=0 MASTER_PORT=8788 MASTER_ADDR=localhost tira-run-inference-server --script /predict.py --port 8787"
```

