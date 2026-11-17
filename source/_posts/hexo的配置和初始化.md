---
title: hexo的配置和初始化
date: 2025-08-25 13:22:14
tags: [hexo, 学习]
cover: /img/hexo.jpg
description: 项目启动初始化配置和三个hexo常用指令
---

## 0.首先安装好node.js和git（安装基础环境）

## 1.安装hexo
``` bash
$ npm install -g hexo-cli
```
### 初始化hexo
``` bash
$ hexo init
```
### 安装hexo依赖
``` bash
$ npm install
```
## 2.现在就可以通过hexo命令预览初创hexo网站了！
### 清除hexo缓存
``` bash
$ hexo cl
```
### 构建hexo
``` bash
$ hexo g
```
### 本地预览hexo
``` bash
$ hexo s
```
### 上传hexo
``` bash
$ hexo d
```