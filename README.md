# Team hierocles-of-alexandria for ValueEval'24

## Dockerization of Prediction Model
- uses model from [https://huggingface.co/SotirisLegkas/multi-head-xlm-xl-tokens-38](https://huggingface.co/SotirisLegkas/multi-head-xlm-xl-tokens-38)
```bash
# build
docker build -f Dockerfile -t valueeval24-hierocles-of-alexandria:1.0.0 .

# run
tira-run \
  --input-directory "$PWD/valueeval24/test"\
  --output-directory "$PWD/output" \
  --image valueeval24-hierocles-of-alexandria:1.0.0

# or
docker run --rm \
  -v "$PWD/valueeval24/test:/dataset" -v "$PWD/output:/output" \
  valueeval24-hierocles-of-alexandria:1.0.0

# view results
cat output/run.tsv
```

## Inference Server
```bash
PORT=8001

docker run --rm -it --init \
  -v "$PWD/logs:/workspace/logs" \
  -p $PORT:$PORT \
  --entrypoint tira-run-inference-server
  valueeval24-hierocles-of-alexandria:1.0.0 \
  --script /predict.py --port $PORT
```
