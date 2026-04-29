# JMeter load tests

The plan exposes the following Java properties (set with `-J`):

| Property      | Default                          | Meaning                                  |
| ------------- | -------------------------------- | ---------------------------------------- |
| `host`        | `localhost`                      | API hostname                             |
| `port`        | `7860`                           | API port                                 |
| `scheme`      | `http`                           | `http` for local, `https` for HF Spaces  |
| `threads`     | `50`                             | concurrent virtual users                 |
| `rampup`      | `20`                             | ramp-up duration in seconds              |
| `duration`    | `120`                            | test duration in seconds                 |
| `image_path`  | `tests/fixtures/tennis.jpg`      | JPEG/PNG to upload                       |
| `mime`        | `image/jpeg`                     | MIME type matching `image_path`          |

### Local (Docker)

```bash
mkdir -p jmeter/results jmeter/report
jmeter -n -t jmeter/load_test.jmx \
       -l jmeter/results/result.jtl \
       -e -o jmeter/report \
       -Jhost=localhost -Jport=7860 \
       -Jthreads=50 -Jrampup=20 -Jduration=120 \
       -Jimage_path=tests/fixtures/tennis.jpg
```

### Cloud (Hugging Face Spaces)

```bash
mkdir -p jmeter/results jmeter/report
jmeter -n -t jmeter/load_test.jmx \
       -l jmeter/results/cloud.jtl \
       -e -o jmeter/report-cloud \
       -Jhost=<your-username>-sports-vit-api.hf.space \
       -Jport=443 -Jscheme=https \
       -Jthreads=20 -Jrampup=30 -Jduration=180 \
       -Jimage_path=tests/fixtures/tennis.jpg
```

The HTML dashboard at `jmeter/report/index.html` (or `report-cloud/`)
contains throughput (TPS), response-time percentiles (incl. P95 / P99)
and the active-thread timeline.
