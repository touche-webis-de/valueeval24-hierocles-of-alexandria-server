# Team hierocles-of-alexandria for ValueEval'24

## Dockerization of Prediction Model
- uses model from [https://huggingface.co/SotirisLegkas/multi-head-xlm-xl-tokens-38](https://huggingface.co/SotirisLegkas/multi-head-xlm-xl-tokens-38)
```bash
# build
docker build -f Dockerfile -t valueeval24-hierocles-of-alexandria:1.0.0 .

# run
docker run --rm \
  -v "$PWD/valueeval24/test:/dataset" -v "$PWD/output:/output" \
  valueeval24-hierocles-of-alexandria:1.0.0

# view results
cat output/run.tsv
```

