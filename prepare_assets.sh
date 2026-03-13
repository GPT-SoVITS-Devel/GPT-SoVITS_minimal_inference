#!/usr/bin/env bash
set -e

ASSETS_DIR=assets
mkdir -p $ASSETS_DIR

################################
# Download pretrained_models.zip
################################
echo "Downloading pretrained_models.zip..."
wget -q -O $ASSETS_DIR/pretrained_models.zip \
    https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip

echo "Extracting selected pretrained models..."
mkdir -p tmp
unzip -q $ASSETS_DIR/pretrained_models.zip -d tmp

# 指定需要保留的模型
MODELS=(
    "chinese-hubert-base"
    "chinese-roberta-wwm-ext-large"
    "fast_langdetect"
    "sv"
)

# 创建最终目标目录
FINAL_DIR=$ASSETS_DIR/pretrained_models
mkdir -p $FINAL_DIR

# 移动需要的模型，忽略不存在的
for model in "${MODELS[@]}"; do
    if [ -d "tmp/pretrained_models/$model" ]; then
        mv "tmp/pretrained_models/$model" "$FINAL_DIR/"
    fi
done

rm -rf tmp
rm $ASSETS_DIR/pretrained_models.zip

################################
# Download G2PWModel.zip
################################
echo "Downloading G2PWModel.zip..."
wget -q -O $ASSETS_DIR/G2PWModel.zip \
    https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip

echo "Extracting G2PWModel.zip..."
unzip -q $ASSETS_DIR/G2PWModel.zip -d $ASSETS_DIR
rm $ASSETS_DIR/G2PWModel.zip

################################
# Download nltk_data.zip
################################
echo "Downloading nltk_data.zip..."
wget -q -O $ASSETS_DIR/nltk_data.zip \
    https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip

echo "Extracting nltk_data.zip..."
unzip -q $ASSETS_DIR/nltk_data.zip -d $ASSETS_DIR
rm $ASSETS_DIR/nltk_data.zip


echo "Assets ready:"
find $ASSETS_DIR -maxdepth 3