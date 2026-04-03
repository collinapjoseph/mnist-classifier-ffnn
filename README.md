# Feedforward Neural Network MNIST Classifier
Generated using Claude Sonnet 4.6<br>

Requirements: Pytorch, Onnx, Flask, Numpy, Matplotlib, Pillow<br>

**TO RUN:**<br>
1. In Docker terminal, navigate to project folder or pull image from [dockerhub/collinapjoseph/mnist-classifier](https://hub.docker.com/repository/docker/collinapjoseph/mnist-classifier/general)
2. `docker build -t mnist-classifier`
3. `docker run -p 5000:5000 mnist-classifier`
4. Open browser and navigate to `http://localhost:5000` or click the link in Docker

**ALTERNATIVELY:** 
1. Install dependencies
2. Set environment variables: <br>
```
ENV HOST=0.0.0.0`
ENV PORT=5000
```
3. Start server: <br>
```
python server.py
```
4. Open browser and navigate to `http://localhost:5000`
