TO RUN:
(1) In Docker terminal, navigate to project folder. (e.g. cd ~/Downloads/mnist-pipeline)
(2) docker build -t mnist-classifier .
(3) docker run -p 5000:5000 mnist-classifier
(4) Open browser and navigate to http://localhost:5000 or click the link in Docker