## BUILD
```bash
docker build -f Dockerfile -t aerostack2/rl_exploration:latest .
```

## RUN
```bash
xhost +
docker run --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY aerostack2/rl_exploration
```

## COMPOSE
```bash
xhost +
docker compose up -d && docker exec -it as2_rl_exploration bash
```