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
sumot parallel \
--db-host 192.168.0.171 --record --keep-container --max-parallel 20 \
--epoch-count 2000 \
--name QMAP03 \
--out-dir ~/tmp/sumosim/results/results-2025-02-QMAP03 \
--parallel-config q-map-2
```

Test at ben
```
sumot parallel -n QMAP01 \
--parallel-config q-map-0 -e 20000 \
--db-host ben.local \
--record --keep-container \
--out-dir ~/tmp/sumosim/results/results-2025-02-QMAP01 --max-parallel 15
```

Create reports
- Define report in report.yml and enumdescs.yml
```
sumot report \
-d $HOME/tmp/sumosim/results/ \
-o $HOME/tmp/sumosim/reports/r10
```
- If videos are created, delete the images from results/q-heat-maps
- Zip the results directory and store it on Google Drive

Upload (rsync) the reports to entelijan
`TODO cmd form ben`

