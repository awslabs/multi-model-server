# sockeye-serving
This example shows how to serve Sockeye models for machine translation.
The custom handler is implemented in `sockeye_service.py`.
Since Sockeye has many dependencies, it's convenient to use Docker.
For simplicity, we'll use a pre-trained model and make some assumptions about how we preprocess the data.

## Getting Started With Docker
Pull the latest Docker image:
```bash
docker pull jwoo11/sockeye-serving
```

Download the example model archive file (MAR).
This is a ZIP archive containing the parameter files and scripts needed to run translation for a particular language:
* https://www.dropbox.com/s/pk7hmp7a5zjcfcj/zh.mar?dl=0

Extract the MAR file to `/tmp/models`.
 We'll use this directory as a bind mount for Docker:
```bash
unzip -d /tmp/models/zh zh.mar
```

Start the server:
```bash
docker run -itd --name mms -p 8080:8080 -p 8081:8081 -v /tmp/models/:/models jwoo11/sockeye-serving serve
```

Try making some requests:
```bash
# load the model
curl -X POST "http://localhost:8081/models?synchronous=true&initial_workers=1&url=zh.mar"

# show the status of the ZH model
curl -X GET "http://localhost:8081/models/zh"

# translate a sentence
curl -X POST "http://localhost:8080/predictions/zh" -H "Content-Type: application/json" \
    -d '{ "text": "我的世界是一款開放世界遊戲，玩家沒有具體要完成的目標，即玩家有超高的自由度選擇如何玩遊戲" }'
```

For more information on MAR files and the built-in REST APIs, see:
* https://github.com/awslabs/mxnet-model-server/tree/master/docs
