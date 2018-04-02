# Serverless Inference with MMS on FARGATE

This is self-contained step by step guide that shows how to create ECS with Fargate in order to do a serverless inference with MMS. Even though it is fully self contained we do expect reader to have some knowledge about the following topics:

* [MMS](https://github.com/awslabs/mxnet-model-server)
* [ECS](https://aws.amazon.com/ecs)
* [Fargate](https://aws.amazon.com/fargate)
* [MXNet](https://aws.amazon.com/mxnet/)
* [Docker](https://www.docker.com/)

Since we are doing inference, we need to have a pre-trained model that we can use to run inference. For the sake of this article we will be using [SqueezeNet model](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md#squeezenet_v1.1). In short SqueezeNet is the model that allows you to recognize objects on the picture. 

Now, when we have the model chosen let's discuss on the high level how our end solution will looks like:

![architecture](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/overal_architecture.png)

Do not be worried if something is not clear from the picture. We are going to walk you step by step. And these steps are:

1. Familiarize yourself with our containers
2. Create a SqueezeNet task definition (with the docker container of MMS) 
3. Create Fargate ECS cluster
4. Create Application Load Balancer
5. Create Squeezenet Fargate service on the cluster
6. Profit!

Let the show begin...

## Familiarize Yourself With Our Containers 

With our current release of [MMS, 0.3](https://github.com/awslabs/mxnet-model-server/releases/tag/v0.3.0), we now providing official containers with the MMS preinstalled already. There are 2 containers:

* [awsdeeplearningteam/mms_cpu](https://hub.docker.com/r/awsdeeplearningteam/mms_cpu/)
* [awsdeeplearningteam/mms_gpu](https://hub.docker.com/r/awsdeeplearningteam/mms_gpu/)

Since we are doing inference we are going to use cpu container(mms_cpu). 

There are several constrains that one should consider when using Fargate:

1. There is no GPU support at the moment
2. mms_cpu container is optimized for the Skylake Intel processors (that we have on our [C5 EC2 instances](https://aws.amazon.com/ec2/instance-types/c5/)). However since we are using Fargate, unfortunately there is no guarantee that the actual hardware will be Skylake  

Our containers comes with pre-installed config of the SqueezeNet model (such a nice coincidence that we have decided to run inference for exactly this model :) ). Even though the config is pre-baked in the container we would highly recommend you to have a quick look on it, just to familiarize yourself, since for your own model most likely you will have to update it, plus it very-very simple, [check it yourself](https://github.com/awslabs/mxnet-model-server/blob/master/docker/mms_app_cpu.conf). To be more precise, here is the [line in the config](https://github.com/awslabs/mxnet-model-server/blob/master/docker/mms_app_cpu.conf#L3) that is pointing to the actual binary of the model. By a close look one can see that it is just a  public HTTPS link to the binary:

```
https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

So there is no need to pre-bake actual binary of the model to the container, you can just specify the HTTPS link to the binary.

The last question that we need to address: how we should be starting our MMS within our container. And the answer is very simple, you just need to set the following ENTRYPOINT ([what is ENTRYPOINT?](https://docs.docker.com/engine/reference/builder/#entrypoint): 

```bash
mxnet-model-server start --mms-config /mxnet-model-server/mms_app_cpu.conf
```

And this it, nothing else.

At this point I think we are ready to start creating actual Task definition.

## Create a SqueezeNet Task Definition

This is the first task where we finally ready to start doing something:

1. Log-in to the AWS console and go to the Elastic Cloud Service / Task Definitions and press “Create new Task Definition”:

![task def](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1_Create_task_definition.png)

2. Now you need to specify type of the task, surprise-surprise, we will be using the Fargate task:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/2_Select_Fargate.png)

3. Task requires some configuration, let's look on it step by step, first set the name:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3_Config_1.png)

Now is important part, one need to create [IAM role](https://aws.amazon.com/iam) that will be used to publish metrics to CloudWatch:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/Task+Execution+IAM+Role+.png)

Our containers are optimized for 8 vCPUs, however in our example we going to use slightly smaller task with 4 vCPUs and 8 GB of RAM:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/cpu+and+ram.png)

2. Now it is time to configure the actual container that the task should be executing

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/container+step+1.png)

3. Next one need to specify port mapping, we need to expose port 8080. This is the port that our container is listening to. If needed it can be configured via the config [here](https://github.com/awslabs/mxnet-model-server/blob/master/docker/mms_app_cpu.conf#L40).

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/port+8080.png)

Now the health checks. MMS has pre-configured endpoint /ping that can be used for health checks.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/container+health+checks.png)

and the environment, with the entry point that we have discussed earlier:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/entrypoint.png)

Everything else can be left as default. So feel free to press add and finish the creation of our very first task. If everything is ok you should now be able to see your task in the list of task definitions.

Finally we have a task that we can be executed, but we need something that will be in charge of the actual execution. This something, in terms of ECS, is called Service. Service is capable of running multiple tasks and in charge of making sure that required amount of tasks is always running, restarting un-health tasks adding more tasks when needed, etc. If the service is going to be accessible from the Internet, we first need to configure load balancer that will  be in charge of serving the traffic from the Internet and redirecting it to the tasks. So we are going to create an LB now:

## Create LoadBalancer

AWS Supports several different types of Load Blanacers:


* Application Load Balancer, that works on the level 7 of the OSI model (effectively with the HTTP/HTTPS protocols)
* TCP Load Balancer and
* deprecated Elastic Load Balancer, which also used to work on the TCP level

For our cluster we are going to use application load balancer.
1. Open AWS console
2. Goto EC2
3. Goto to “Load balancers” section
4. Create new Load Balancer

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1__Create_Load_Balancer.png)

5. Choose “HTTP/HTTPS”

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/2__HTTP_HTTPS+.png)

6. Set all the required details. Most importantly set the VPC. VPC is very important. Having wrong VPC might cause our LB to not be able to communicate with your tasks. So memorize VPC that you going to use here for later.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3_2_Listeners_and_AZ+.png)

7. Next is security group. This is also important. Your security group should:

* allow inbound connection for port 80 (since this is the port on which LB will be listening on)
* member of the security group should be able to talk to the security group where you planing to have your service. In our case we will use same security group everywhere in order to simplify our life.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/4.+Configure+Security+groups.png)

8. Routing configuration is simple. The thing is here you need to create what is called “target group”. But the thing is, ECS will create required service group later on for you. Therefore now you need to create dummy “target group” that you will delete after the creation of the LB. Yes, I know, there should be option to skip this, but there is not :(

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/5.+Configure+Routing+(DUmmy).png)

9. nothing need to be done on the last two steps. Finish the creation and ...
10. now you are ready to remove dummy listener and target group

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/8__Delete_the_dummy_listener.png)

Finally we are done-done-done with the LB, time to move to the service creation

## Create a ECS Service from the ECS Task Definitions

1. Go to Elastic Container Service → Task Definitions and select the task definitions name. Click on actions and select create service.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1.+Go+to+task+definitions.png)

2.  There are two important things on the first step (apart from naming):

* platform version, it should be set to 1.1.0 and
* number of tasks that the service should maintain as healthy all the time, in our case we will use 3

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/number+of+tasks.png)

3. Now it is time to configure the VPC and the security group. One should use EXACTLY same VPC that was used for the LB (and same subnets!)

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.1+Use+the+existing+VPC+Edit+sg.png)

4. As for the security group, it should be either the same security group as you had for the LB, or the one that LB have access to:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.2+SG+Use+existing.png)

5.  Now we can connect our future service to the existing LB. Select the application load balancer and set the LB name:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.3+Add+load+balancing.png)

6. Now we need to specify which port on the LB our service should be listening on:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.4+Configure+load+blancer.png)

7. We are not going to use service discovery now, so uncheck it:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.5+Next.png)

8. Also for the sake of the simplicity we are not going to use auto-scaling functionality

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.3+Auto+scaling.png)

9. Now we are done-done-done and we can move to the final chapter of our journey:

## Test The Inference

First let's figure out the DNS name of our LB. It should be in EC2 => Load Balancers and click on the LB:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/lb_dns.png)

Now we can run the health checks to verify that it is working:
```bash
➜ ~ curl InfraLb-1624382880.us-east-1.elb.amazonaws.com/ping 

http://infralb-1624382880.us-east-1.elb.amazonaws.com/ping
{
    "health": "healthy!"
}
```
And now we are finally ready to run our inference! Let's download example iamge:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
```

The image:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/kitten.jpg)

And the actual inference:
```bash
➜  ~ curl -X POST InfraLb-1624382880.us-east-1.elb.amazonaws.com/squeezenet/predict -F "data=@kitten.jpg"
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
## Instead of a Conclusion

There are many things that we have not touched that can be very useful, like:


* setting up integration between MMS and CloudWatch
* setting auto scaling funcitonality
* running A/B testing of different versions of the model with the Fargate Deployment concepts
* connecting Route 53 domain name
* etc

Each of the topics above probably requires it is own article, so stay tuned ;)

## Authors

* Viacheslav Kovalevskyi (@b0noi)
* Vamshidhar Dantu  
