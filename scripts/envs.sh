# !/bin/bash

# Envirement Install 环境安装
pip install -r requirements.txt

# Install Wandb / Timm
pip install wandb
pip install timm

# Wandb Log 
# Login Wandb登录
# replace {secret key} with your actual secret key
wandb login {secret key}

# Create Log Directory 创建日志存放文件
mkdir ./logs

