# Serverless Inference with MMS on FARGATE

This is self-contained step by step guide that shows how to create launch and server your deep learning models with MMS in a production setup. 
In this document you will learn how to launch MMS with AWS Fargate, in order to achieve a serverless inference.

## Prerequisites

Even though it is fully self-contained we do expect the reader to have some knowledge about the following topics:

* [MMS](https://github.com/awslabs/mxnet-model-server)
* [What is Amazon Elastic Container Service (ECS)](https://aws.amazon.com/ecs)
* [What is Fargate](https://aws.amazon.com/fargate)
* [What is Docker](https://www.docker.com/) and how to use containers

Since we are doing inference, we need to have a pre-trained model that we can use to run inference. 
For the sake of this article, we will be using 
[SqueezeNet model](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md#squeezenet_v1.1). 
In short, SqueezeNet is a model that allows you to recognize objects in a picture. 

Now that we have the model chosen, let's discuss at a high level what our serverless solution will look like:

![architecture](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/architecture_2.png)

Do not be worried if something is not clear from the picture. We are going to walk you through step by step. And these steps are:

1. Familiarize yourself with MMS containers
2. Create a SqueezeNet task definition (with the docker container of MMS) 
3. Create AWS Fargate cluster
4. Create Application Load Balancer
5. Create Squeezenet Fargate service on the cluster
6. Profit!

Let the show begin...

## Familiarize Yourself With Our Containers 

With the current release of [MMS, 0.3](https://github.com/awslabs/mxnet-model-server/releases/tag/v0.3.0), 
Official pre-configured, optimized container images of MMS are provided on [Docker hub](https://hub.docker.com).

* [awsdeeplearningteam/mms_cpu](https://hub.docker.com/r/awsdeeplearningteam/mms_cpu/)
* [awsdeeplearningteam/mms_gpu](https://hub.docker.com/r/awsdeeplearningteam/mms_gpu/)

In our article we are going to use the official CPU container image.

There are several constraints that one should consider when using Fargate:

1. There is no GPU support at the moment.
2. mms_cpu container is optimized for the Skylake Intel processors (that we have on our [C5 EC2 instances](https://aws.amazon.com/ec2/instance-types/c5/)). However, since we are using Fargate, unfortunately there is no guarantee that the actual hardware will be Skylake.

The official container images come with pre-installed config of the SqueezeNet model. Even though the config is pre-baked in the container, it is highly recommended that you understand all the parameters of the MMS configuration file.
Familiarize yourself with the [MMS configuration](https://github.com/awslabs/mxnet-model-server/blob/master/docker/mms_app_cpu.conf) and [configuring MMS Container docs](https://github.com/awslabs/mxnet-model-server/blob/master/docker/README.md).
When you want to launch and host your custom model, you will have to update this configuration. 
Here is the line in the config that is pointing to the binary of the model:

```
https://github.com/awslabs/mxnet-model-server/blob/master/docker/mms_app_cpu.conf#L3
```

Looking closely, one can see that it is just a public HTTPS link to the binary:

```
https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

So there is no need to pre-bake actual binary of the model to the container. You can just specify the S3 link to the binary.

The last question that we need to address: how we should be starting our MMS within our container. 
And the answer is very simple, you just need to set the following [ENTRYPOINT](https://docs.docker.com/engine/reference/builder/#entrypoint): 

```bash
mxnet-model-server start --mms-config /mxnet_model_server/mms_app_cpu.conf
```

You will now have a running container, ready to serve the models configured in the "mms_app_cpu.conf" mentioned in the ENTRYPOINT above. 

At this point, you are ready to start creating actual task definition.

## Create an AWS Fargate task to serve SqueezeNet model

This is the first step towards getting your own "inference service" up and running in a production setup. 

1. Login to the AWS console and go to the Elastic Cloud Service -> Task Definitions and Click “Create new Task Definition”:

![task def](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1_Create_task_definition.png)

2. Now you need to specify the type of the task, you will be using the Fargate task:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/2_Select_Fargate.png)

3. The task requires some configuration, let's look at it step by step. First set the name:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3_Config_1.png)

Now is important part, you need to create a [IAM role](https://aws.amazon.com/iam) that will be used to publish metrics to CloudWatch:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/Task+Execution+IAM+Role+.png)

The containers are optimized for 8 vCPUs, however in this example you are going to use slightly smaller task with 4 vCPUs and 8 GB of RAM:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/cpu+and+ram.png)

4. Now it is time to configure the actual container that the task should be executing.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/container+step+1.png)
<br></br>
*Note:* If you are using a [custom container](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md#customize-the-containers-to-serve-your-custom-deep-learning-models), make sure to first upload your container to Amazon ECR or Dockerhub and replace the link in this step with the link to your uploaded container.

5. The next task is to specify the port mapping. You need to expose container port 8080. 
This is the port that the MMS application inside the container is listening on. 
If needed it can be configured via the config [here](https://github.com/awslabs/mxnet-model-server/blob/master/docker/mms_app_cpu.conf#L40).

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/port+8080.png)

Next, you will have to configure the health-checks. This is the command that ECS should run to find out whether MMS is running within the container or not. MMS has a pre-configured endpoint `/ping`
that can be used for health checks. Configure ECS to reach that endpoint at `http://127.0.0.1:8080/ping` using the `curl` command as shown below:

```bash
curl, http://127.0.0.1:8080/ping
```

The healthcheck portion of your container configuration should look like the image below:

![](https://s3.amazonaws.com/mxnet-model-server/mms-github-docs/MMS+with+Fargate+Article/add+container+healthcheck.png)

After configuring the health-checks, you can go onto configuring the environment, with the entry point that we have discussed earlier:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/entrypoint.png)

Everything else can be left as default. So feel free to click `Create` to create your very first AWS Fargate-task. 
If everything is ok, you should now be able to see your task in the list of task definitions.

In ECS, `Services` are created to run Tasks. A service is in charge of 
running multiple tasks and making sure the that required number of tasks are always running, 
restarting un-healthy tasks, adding more tasks when needed. 
 
 To have your `inference service` accessible over the Internet, you would need to configure a load-balancer (LB). This LB will  be 
 in charge of serving the traffic from the Internet and redirecting it to these newly created tasks. 
 Let's create an Application Load Balancer now:

## Create a Load Balancer

AWS supports several different types of Load Balancers:

* Application Load Balancer: works on the level 7 of the OSI model (effectively with the HTTP/HTTPS protocols)
* TCP Load Balancer 

For your cluster you are going to use application load balancer.
1. Login to the EC2 Console.
2. Go to the “Load balancers” section.
3. Click on Create new Load Balancer.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1__Create_Load_Balancer.png)

5. Choose “HTTP/HTTPS”

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/2__HTTP_HTTPS+.png)

6. Set all the required details. **Make a note of the VPC of the LB**. This is important since the LB's VPC and the ECS
cluster's VPC need to be same for them to communicate with each other.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3_2_Listeners_and_AZ+.png)

7. Next is configuring the security group. This is also important. Your security group should:

* Allow inbound connections for port 80 (since this is the port on which LB will be listening on)
* LBs security group needs to be added to the AWS Fargate service's security group, so that all the traffic from LB is accepted
by your "inference service". 

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/4.+Configure+Security+groups.png)

8. Routing configuration is simple. Here you need to create a “target group”. 
But, in your case the AWS Fargate service, that you will create later, will automatically create a target group.  
Therefore you will create dummy “target group” that you will delete after the creation of the LB. 

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/5.+Configure+Routing+(DUmmy).png)

9. Nothing needs to be done for the last two steps. `Finish` the creation and ...
10. Now you are ready to remove dummy listener and target group

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/8__Delete_the_dummy_listener.png)

Now that you are `done-done-done` with the Load Balancer creation, lets move onto creating our Serverless inference service.

## Creating an ECS Service to launch our AWS Fargate task

1. Go to Elastic Container Service → Task Definitions and select the task definitions name. Click on actions and select create service.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1.+Go+to+task+definitions.png)

2. There are two important things on the first step (apart from naming):

* Platform version: It should be set to 1.1.0 .
* Number of tasks that the service should maintain as healthy all of the time, for this example you will set this to 3.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/number+of+tasks.png)

3. Now it is time to configure the VPC and the security group. **You should use the same VPC that was used for the LB (and same subnets!).**

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.1+Use+the+existing+VPC+Edit+sg.png)

4. As for the security group, it should be either the same security group as you had for the LB, or the one that accepts traffic from the LBs security group.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.2+SG+Use+existing.png)

5. Now you can connect your service to the LB that was created in the previous section. Select the "Application Load Balancer" and set the LB name:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.3+Add+load+balancing.png)

6. Now you need to specify which port on the LB our service should be listening on:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.4+Configure+load+blancer.png)

7. You are not going to use service discovery now, so uncheck it:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.5+Next.png)

8. In this document, we are not using auto-scaling options. For an actual production system, it is advisable to have this configuration setup.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.3+Auto+scaling.png)

9. Now you are `done-done-done` creating a running service. You can move to the final chapter of the journey, which is testing the service you created. 

## Test your service

First find the DNS name of your LB. It should be in `AWS Console -> Service -> EC2 -> Load Balancers` and click on the LB that you created.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/lb_dns.png)

Now you can run the health checks using this load-balancer public DNS name, to verify that your newly created service is working:

```bash
curl InfraLb-1624382880.us-east-1.elb.amazonaws.com/ping 
```

```text
http://infralb-1624382880.us-east-1.elb.amazonaws.com/ping
{
    "health": "healthy!"
}
```

And now you are finally ready to run our inference! Let's download an example image:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
```

The image:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/kitten.jpg)

The output of this query would be as follows,

```bash
curl -X POST InfraLb-1624382880.us-east-1.elb.amazonaws.com/squeezenet/predict -F "data=@kitten.jpg"
```

```text
{
      "prediction": [
    [
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.8515275120735168
      },
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.09674164652824402
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.03909163549542427
      },
      {
        "class": "n02128385 leopard, Panthera pardus",
        "probability": 0.006105933338403702
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.003104303264990449
      }
    ]
  ]
}
```

## Customize the containers to serve your custom deep learning models

For this section, we will show you how you can wrap the provided container images with a custom configuration. 
The custom MMS configuration will contain your custom deep learning models, modified serving workers to suite your needs and any other customizations that you would require to run your "containerized inference service".

To launch a service in AWS Fargate, you will need to have a container image of MMS. Let's look at how you could build a custom container.

1. Create a directory to for your custom-container image work and work from that directory.
```bash
mkdir /tmp/custom_container
```
```bash
cd /tmp/custom_container
```

2. Download the [MMS configuration](https://s3.amazonaws.com/mms-config/mms_app_cpu.conf) for CPU.
```bash
wget https://s3.amazonaws.com/mms-config/mms_app_cpu.conf
```
Example CPU and GPU configurations are available on S3 at [CPU](https://s3.amazonaws.com/mms-config/mms_app_cpu.conf) and [GPU](https://s3.amazonaws.com/mms-config/mms_app_gpu.conf).
Modify the downloaded configuration file and update the `--models` parameter, with your custom model that you want to serve. 
For more on how to modify this, check [Advanced Settings](../docker/advanced_settings.md)

3. For this example, let's edit the downloaded configuration file and add "Resnet-18" model to the list of served models. You will make the following changes to this configuration file. You now have a MMS configuration, which can serve both "SqueezeNet" and "Resnet" models.

```bash
vi mms_app_cpu.conf
```
```text
[MMS Arguments]

# Models names must be seperated with space
--models
squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model resnet=https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model

--service
optional

--gen-api
optional
...
```

4. Write a short Dockerfile which can be used to create your custom MMS container image.
```bash
vi Dockerfile
```
Paste the following in that Dockerfile and save the file.
```text
FROM awsdeeplearningteam/mms_cpu
COPY mms_app_cpu.conf /mxnet_model_server/mms_app_cpu.conf
```

5. Build your container image using the Docker CLI
```bash
docker build -t custom_mms .
```
Verify that the your newly built container image was successfully built, by running the following command. 
```bash
docker images
```

6. You now have a custom container image which can be uploaded to [Amazon ECR Repository](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html) or you can host the container image on [Docker hub registry](hub.docker.com).

This newly created container image can be used for launching your new serverless inference service. Follow the steps mentioned in the previous sections of this document
to create a Amazon Fargate service. 
  

## Instead of a Conclusion

There are a few things that we have not covered here and which are very useful, such as:

* How to set up IAM policies to cloudwatch metrics.
* How to configure auto-scaling on our ECS cluster.
* Running A/B testing of different versions of the model with the Fargate Deployment concepts.

Each of the above topics require their own articles, so stay tuned!!

## Authors

* Aaron Markham
* Vamshidhar Dantu 
* Viacheslav Kovalevskyi (@b0noi) 
