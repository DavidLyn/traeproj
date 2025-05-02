# 语音助手项目说明文档

## 项目概述

这是一个基于FastAPI的语音助手应用，支持通过WebSocket进行实时语音识别，并调用阿里云通义千问大模型进行对话。该应用使用Whisper模型进行语音识别，并通过OpenAI兼容接口调用通义千问大模型进行回复。

## 功能特点

- 实时语音识别：使用OpenAI的Whisper模型进行语音转文本
- 大模型对话：通过OpenAI SDK调用阿里云通义千问大模型
- 流式响应：支持大模型的流式输出，提高响应速度
- WebSocket通信：实现浏览器与服务器之间的实时通信
- 用户友好界面：简洁的前端界面，支持语音输入和显示对话历史
- 日志记录：详细记录系统运行状态和token使用情况

## 技术栈

- 后端：FastAPI、Python 3.10+
- 语音识别：OpenAI Whisper
- 大模型：阿里云通义千问（通过OpenAI兼容接口）
- 前端：HTML、JavaScript、WebRTC
- 通信：WebSocket

## 安装与配置

### 环境要求

- Python 3.10 或更高版本
- 支持WebRTC的现代浏览器

### 安装依赖

```bash
pip install fastapi uvicorn openai python-dotenv whisper
```

### 配置环境变量

在项目根目录创建`.env`文件，添加以下内容：

```
TONGYI_API_KEY=您的通义千问API密钥
TONGYI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 使用方法

### 启动服务器

```bash
python main.py
```

服务器将在`http://127.0.0.1:8000`上运行。

### 访问应用

在浏览器中访问`http://127.0.0.1:8000/client`，即可打开语音助手界面。

### 使用流程

1. 点击"开始录音"按钮，授予麦克风权限
2. 对着麦克风说话
3. 系统会自动识别您的语音并发送给大模型
4. 大模型的回复会显示在界面上
5. 点击"停止录音"按钮结束对话

## 项目结构

- `main.py`：主程序文件，包含FastAPI应用和所有功能实现
- `.env`：环境变量配置文件
- `.gitignore`：Git忽略文件配置
- `app.log`：应用日志文件

## 注意事项

- 通义千问大模型仅支持流式模式，必须设置`stream=True`参数
- Whisper模型默认使用"tiny"版本，可以根据需要在代码中修改为其他版本
- 首次运行时会下载Whisper模型，可能需要一些时间
- 确保麦克风设备正常工作并已授权给浏览器使用

## 故障排除

- 如果遇到语音识别问题，请检查麦克风设置和浏览器权限
- 如果大模型调用失败，请检查API密钥和网络连接
- 查看`app.log`文件获取详细的错误信息和运行日志

## 贡献指南

欢迎提交问题报告和改进建议。如需贡献代码，请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支
3. 提交您的更改
4. 推送到您的分支
5. 创建Pull Request

## 许可证

[MIT License](LICENSE)
