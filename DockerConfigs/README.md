# Build Docker to run Deep Feature Flow
Build Image use our Dockerfile configuration
```bash
cd ./MXNet
docker build -t mxnet_dff/python:gpu -f ./Dockerfile.python.gpu .
```
- `-t` is for tag
- `-f` is for dockerfile path
 