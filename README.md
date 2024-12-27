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

### Create a video from images
`ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p q-values-002.mp4`

### Docker 

TODO Docker

 * config host and port for database. db_host, db_port . host -> sim_host, sim_port
 * use local user in docker sumot. 
  * create necesaaey directories during build and set to 775

```
docker network create sumo01
docker run --rm --name sumo1 --network sumot01 -v $HOME/tmp/sumosim/q/docker:/root/tmp/sumosim/q sumo sumo udp --port 4401
docker run  -e PYTHONUNBUFFERED=True --network sumot01 \
-v $HOME/tmp/sumosim/q/docker:/root/tmp/sumosim/q \
sumot uv run sumot qtrain -n D01 --auto-naming -e 500 -p 4401 -h sumo1



docker network create sumo02
docker run --rm --name sumo2 --network sumot02 sumo sumo udp --port 4402
docker run -e PYTHONUNBUFFERED=True --network sumot02 \
-v $HOME/tmp/sumosim/q/docker:/root/tmp/sumosim/q \
sumot uv run sumot qtrain -n D02 --auto-naming -e 2000 -p 4402 -h sumo2


--user $(id -u):$(id -g)
```
