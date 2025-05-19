
## Демонстрация работы нашего решения

### Установка игры
```
mamba create -n "lux-s3" "python==3.11"
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3/
pip install -e Lux-Design-S3/src
```

### Запуск битвы бота с самим собой на случайной карте
```
luxai-s3 repo_path\Kaggle_luxai\lux_bot\main.py repo_path\Kaggle_luxai\lux_bot\main.py  --output replay.html
```

Визуализацию матча можно посмотреть в сгенерированном файле `replay.html`