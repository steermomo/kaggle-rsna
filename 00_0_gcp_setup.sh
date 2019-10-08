# https://course.fast.ai/start_gcp.html

# 命令根据[fastai GCP setup](https://course.fast.ai/start_gcp.html), 并针对我的任务进行了修改。

# Google Cloud’s command line interface (CLI)安装配置部分见[fastai GCP setup](https://course.fast.ai/start_gcp.html)

# 另外因为火星网络环境不够稳定, 我在wsl中配置了proxychains4, 连到Windows主机的sock5代理.


# 创建抢占式实例
```bash
export IMAGE_FAMILY="pytorch-latest-gpu" # or "pytorch-latest-cpu" for non-GPU instances
export ZONE="asia-southeast1-b" # budget: "us-west1-b"
export INSTANCE_NAME="kaggle-rsna"
export INSTANCE_TYPE="n1-highmem-8" # budget: "n1-highmem-4"

# budget: 'type=nvidia-tesla-k80,count=1'
proxychains4 gcloud compute instances create $INSTANCE_NAME --zone=$ZONE --image-family=$IMAGE_FAMILY --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE  --accelerator="type=nvidia-tesla-p4,count=1"  --machine-type=$INSTANCE_TYPE  --boot-disk-size=800GB --metadata="install-nvidia-driver=True" --preemptible
```


#创建完成之后, ssh连接上去并开启端口转发, 创建的主机默认有`jupyter`用户, 并且在`8080`端口开了jupyter lab. 这里就用`jupyter`用户连上去, 并开启本地端口转发.

```bash
proxychains4 gcloud beta compute --project "limongty" ssh --zone "asia-southeast1-b" jupyter@"kaggle-rsna" -- -L 8080:localhost:8080
```

# 连接成功后本地浏览器打开`localhost:8080`可以使用远程环境.

# 网络不稳定， 必须开tmux