sds
===
A very hacky **S**table **D**ifussion **S**erving 

## Installation

Using virtualenv is highly recommended.

1. `pip install -r requirements.txt`
2. Run `fetch.sh` and `xformers.sh` (takes a long time) in parallel

## Usage

### Standalone server

Start server. If `--addr` isn't provided, default to `localhost:9001`.

```
python server.py --addr <addr>:<port>
```

Run inference, results will be saved to current directory as JPEG files.

```
curl -vJO -d "prompt=" http://<addr>:<port>/sd
```

### Scheduler + Worker
TODO

## LICENSE
MIT