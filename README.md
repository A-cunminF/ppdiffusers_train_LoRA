# Dreambooth和LoRA训练脚本

## 来源
本脚本修改自：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/dreambooth 中的两个官方脚本，对pytorch2paddle功能进行了完善

本脚本的使用方式与官方脚本大体一样，仅有两个新增加的参数传入需要注意

#### 官方脚本的问题
1. 目前官方脚本仅支持从BOS下载paddle模型
2. FROM_DIFFUSERS、TO_DIFFUSERS和FROM_HF_HUB三个环境变量无法对该脚本生效，因此无法读取、保存pytorch diffusers模型
3. PaddleNLP的各类Tokenizer存在bug，无法识别HF的subfolder

具体详见这两个github issue：https://github.com/PaddlePaddle/PaddleNLP/issues/5847 和 https://github.com/PaddlePaddle/PaddleNLP/issues/5868

#### 本脚本解决的问题
1. 在原脚本基础上，增加from_diffusers和to_diffusers两个参数传入：from_diffusers激活时可读取pytorch diffusers模型，to_diffusers激活时可保存pytorch safetensors模型

#### 暂时需要等待修复才能解决的问题
1. 需要等待PaddleNLP各类Tokenizer对huggingface hub subfolder的读取问题修复才能解决从huggingface下载模型错误的问题

具体参考：https://github.com/PaddlePaddle/PaddleNLP/pull/5871

## 使用方式

#### 脚本需要修改的参数介绍

--from_diffusers：读取pytorch diffusers模型

--to_diffusers：保存成pytorch safetensors模型

脚本的其他参数请参考：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/examples/dreambooth

#### Step1：配置环境

安装ppdiffusers及其依赖包，在notebook中运行

```shell
!python -m pip install --upgrade ppdiffusers --user
```
或者可以在终端里运行
```shell
python -m pip install --upgrade ppdiffusers --user
```

#### Step2：上传/下载你的模型


此处以https://huggingface.co/stablediffusionapi/anything-v5 中的模型为例进行介绍，此模型是HF_HUB上的民间模型，目前BOS上没有镜像。在huggingface上保证你找到的模型是diffusers模型，一个diffusers模型的典型目录是这样的，必须要有model_index.json文件，且有这些目录：

![](https://ai-studio-static-online.cdn.bcebos.com/c4fc0a3049004ef1a3396addb1efb542cbdd05ac5b7249ae84b0fdf72a32c3bc)

如果你需要的模型没有这些，则可以使用https://github.com/PaddlePaddle/PaddleNLP/tree/develop/ppdiffusers/scripts/convert_diffusers_model 的脚本对你需要的模型进行转换。一般而言只要不是太冷门的模型huggingface上都会有转换好的版本。

在AI Studio中打开终端，使用如下命令下载stablediffusionapi/anything-v5模型：
```shell
git lfs install
cd work
git clone https://huggingface.co/stablediffusionapi/anything-v5
```

考虑到AI Studio下载huggingface hub上的模型速度比较慢，因此推荐先下载模型到本地，再打压缩包上传，在终端内解压。特别注意：AI Studio上传单个文件最大限制500M，因此对大模型而言需要分卷打压缩包，假设压缩包名称为：
```
anything-v5.zip.001
anything-v5.zip.002
...
anything-v5.zip.010
```

则在终端内可以用以下命令先对压缩包合并再解压：
```shell
cat anything-v5.zip.* > anything-v5.zip
unzip anything-v5.zip
```

#### Step3：上传你需要用于训练的图片

这里以Sks Dog 训练教程为例，在train_data目录下放了五张小狗图片。考虑到许多使用本项目的人是从AI绘图圈来的小白，所以，重要的事情说三遍：

**不需要对每一张小狗的图片进行单独打标，只需要设置好instance_prompt即可！不需要对图片进行裁切，只需要设置好resolution即可！**

**不需要对每一张小狗的图片进行单独打标，只需要设置好instance_prompt即可！不需要对图片进行裁切，只需要设置好resolution即可！**

**不需要对每一张小狗的图片进行单独打标，只需要设置好instance_prompt即可！不需要对图片进行裁切，只需要设置好resolution即可！**

#### Step4：开始训练

**如果你要完整训练Dreambooth模型，且需要读取和保存pytorch模型的，请使参考如下命令，注意，这条命令会占用22G左右的显存，请使用能提供足够显存的GPU环境**
```shell
python train_dreambooth_ver_paddle.py \
  --pretrained_model_name_or_path="./work/anything-v5"  \
  --instance_data_dir="./train_data" \
  --output_dir="dreambooth_outputs" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --from_diffusers \
  --to_diffusers
```

运行效果如图

![](https://ai-studio-static-online.cdn.bcebos.com/6cd05a0d41e04cb4945083db34d18bd58feb6d74c73841669ad9aacfb4ea1352)

运行结果如图

![](https://ai-studio-static-online.cdn.bcebos.com/01f0c35ec36f49ffb8425ae59ff06b9f7fdb3b1b329643b7acff575d62eed0fe)

![](https://ai-studio-static-online.cdn.bcebos.com/fa972f7df3534fff8b85258804ed53f9efe32f12f5b04d98bba9b2e7b7dc7a54)

**如果你要训练LoRA模型，且需要读取和保存pytorch模型的，请参考如下命令**

```shell
python train_dreambooth_lora_ver_paddle.py \
  --pretrained_model_name_or_path="./work/anything-v5"  \
  --instance_data_dir="./train_data" \
  --output_dir="lora_outputs" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --lora_rank=4 \
  --seed=0 \
  --from_diffusers \
  --to_diffusers
```

运行结果如图

![](https://ai-studio-static-online.cdn.bcebos.com/dcc34cd12a93458eac27413b3e2cfdae009336e3e5ca4179b035dbb61fbf5dbf)

## 在AI Studio中使用训练好的Dreambooth或LoRA模型

#### Dreambooth模型
```python
from ppdiffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import paddle

pipe = DiffusionPipeline.from_pretrained("./dreambooth_outputs",from_diffusers=True)

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=50).images[0]
image
```

#### LoRA模型
```python
from ppdiffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import paddle

pipe = DiffusionPipeline.from_pretrained("./work/anything-v5",from_diffusers=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.load_attn_procs("./lora_outputs/pytorch_lora_weights.safetensors",from_diffusers=True)

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=50).images[0]
image
```
