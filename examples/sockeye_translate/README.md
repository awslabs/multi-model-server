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

Create a file called `config.properties` under `/tmp/models`. We'll use this directory as a bind mount:
```properties
vmargs=-Xmx128m -XX:-UseLargePages -XX:+UseG1GC -XX:MaxMetaspaceSize=32M -XX:MaxDirectMemorySize=10m -XX:+ExitOnOutOfMemoryError
model_store=/models
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
```

Start the server:
```bash
docker run -itd --name mms -p 8080:8080  -p 8081:8081 -v /tmp/models/:/models jwoo11/sockeye-serving \
    mxnet-model-server --start --mms-config /models/config.properties
```

Try making some requests using a remote model:
```bash
# URL of a remote machine translation model
URL="https://www.dropbox.com/s/pk7hmp7a5zjcfcj/zh.mar?dl=1"

until curl -X POST "http://localhost:8081/models?url=${URL}"
do
  echo "Waiting for initialization..."
  sleep 1
done

# set the number of workers to 1
curl -X PUT "http://localhost:8081/models/zh?min_worker=1"

# show the status of the ZH model
curl -X GET "http://localhost:8081/models/zh"

# translate a sentence
curl -X POST "http://localhost:8080/predictions/zh" -H "Content-Type: application/json" \
    -d '{ "text": "我的世界是一款開放世界遊戲，玩家沒有具體要完成的目標，即玩家有超高的自由度選擇如何玩遊戲" }'
```

## Loading a Local Model
Download the example model archive file (MAR):
* https://www.dropbox.com/s/pk7hmp7a5zjcfcj/zh.mar?dl=0

Move the MAR file to `/tmp/models`. You'll still need `config.properties` to reside in the same directory. 

Start the server as before. You may need to stop and remove the container from a previous run.

Load the local model and try some requests. You'll have to unload the model first, if it's already loaded:
```bash
curl -X DELETE "http://localhost:8081/models/zh"
curl -X POST "http://localhost:8081/models?url=zh.mar"
curl -X PUT "http://localhost:8081/models/zh?min_worker=1"
```

It's also possible to use the extracted MAR file as the URL:
```bash
unzip -d /tmp/models/zh zh.mar
curl -X POST "http://localhost:8081/models?url=zh"
```

For more information on MAR files and the built-in REST APIs, see:
* https://github.com/awslabs/mxnet-model-server/tree/master/docs
