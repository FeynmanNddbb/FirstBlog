---
title: butterfly的安装和配置
date: 2025-08-25 13:32:05
tags: [学习, butterfly]
cover: /img/butterfly.jpg
description: 主题的安装、升级和项配置建议
---
## 安装主题
### 在你的 Hexo 根目录
``` bash
npm install hexo-theme-butterfly
```
### 升級方法：在 Hexo 根目录下，运行 
``` bash
npm update hexo-theme-butterfly
```
## 应用主题
### 修改 Hexo 根目錄下的 _config.yml，把主題改為 butterfly 
```
theme: butterfly
```

## 安装插件
### 如果你沒有 pug 以及 stylus 的渲染器，請下載安裝：
``` bash
npm install hexo-renderer-pug hexo-renderer-stylus --save
```