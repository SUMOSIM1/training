## Training for sumosim robots

### Getting started

#### Prerequisits
* uv
* docker and docker compose

##### Start a database

In SUMOSIM/sumosim run: 
```
docker compose up -d
```

##### Start a sumosim simulator

For details see (sumosim simulator)[https://github.com/SUMOSIM1/sumosim/tree/reinforcement]

```
sumosim start --port [PORTNUMBER]

e.g

sumosim start --port 5555
```
### Running
`uv run sumot --help`

### Formating
in project root call: `ruff format`

### Linting
in project root call: `ruff check` or `ruff check --fix`

### Build a docker image

For work `docker buildx build --build-arg DOCKERREG=dockerregistry.seu.test.sozvers.at/ -t sumot .`

Elsewhere `docker buildx build -t sumot .`

### Trainings

#### Cross validation

TODO

#### Mapping

Test at work
```
sumot parallel -n M04 \
--parallel-config q-map-0 -e 1000 \
--db-host 10.133.20.219 \
--record \
--out-dir ~/tmp/sumosim/test/results/m04 --max-parallel 15
```


```
sumot parallel -n QMAP01 \
--parallel-config q-map-0 -e 20000 \
--db-host ben.local \
--record --keep-container \
--out-dir ~/tmp/sumosim/results/results-2025-02-QMAP01 --max-parallel 15
```

```
sumot report \
-d $HOME/tmp/sumosim/results-2025-01/results \
-d $HOME/tmp/sumosim/results-2025-003 -d $HOME/tmp/sumosim/results-2025-01-002 \
-d $HOME/tmp/sumosim/results-2025-02-Q-CV5/ \
-d $HOME/tmp/sumosim/results/results-2025-02-QMAP01 \
-d $HOME/tmp/sumosim/results-2025-02-Q-CV7 \
-o $HOME/tmp/sumosim/reports/r4 
```

