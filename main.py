from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import whisper
import tempfile
import os
import sys  # 添加sys模块导入
import asyncio
import numpy as np
import io
import wave
import json
import logging  # 导入日志模块
import time  # 导入时间模块
from openai import OpenAI  # 导入OpenAI SDK
from dotenv import load_dotenv  # 导入dotenv库

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("语音助手")

# 加载.env文件中的环境变量
load_dotenv()

# 获取通义千问API密钥
tongyi_api_key = os.getenv("TONGYI_API_KEY")
if not tongyi_api_key:
    logger.warning("警告: 未设置通义千问API密钥，请在.env文件中配置TONGYI_API_KEY")

# 获取通义千问API地址
tongyi_api_base = os.getenv("TONGYI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

app = FastAPI()

# 加载Whisper模型（可以选择不同大小的模型：tiny, base, small, medium, large）
# 较小的模型速度更快但准确性较低，较大的模型准确性更高但需要更多资源
# model = whisper.load_model("base")
model = whisper.load_model("tiny")

# 初始化OpenAI客户端
client = OpenAI(
    api_key=tongyi_api_key,
    base_url=tongyi_api_base
)

# 使用OpenAI SDK调用通义千问大模型
async def call_tongyi_model(prompt):
    """
    使用OpenAI SDK调用通义千问大模型 (流式)

    参数:
        prompt (str): 用户输入的文本

    返回:
        str: 模型生成的完整回复
    """
    try:
        logger.info(f"调用通义千问大模型 (流式)，输入: '{prompt}'")
        start_time = time.time()

        # 使用异步执行模型调用
        loop = asyncio.get_event_loop()

        # 使用OpenAI SDK调用模型，启用流式输出
        def call_api_stream():
            stream = client.chat.completions.create(
                model="qwen3-235b-a22b",  # 确认使用的模型名称，日志中是qwen-max
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True  # 启用流式模式
            )
            return stream

        stream = await loop.run_in_executor(None, call_api_stream)

        full_reply = ""
        completion_tokens = 0
        prompt_tokens = 0 # Prompt tokens 通常在第一个 chunk 或 usage 中提供，这里简化处理

        logger.info("开始接收流式响应...")
        for chunk in stream:
            # 提取 token 使用信息 (如果可用)
            # 注意：通义千问的流式接口可能不会在每个 chunk 中都提供完整的 usage 信息
            # 通常在最后一个 chunk 或需要单独处理
            if hasattr(chunk, 'usage') and chunk.usage:
                 if chunk.usage.prompt_tokens:
                     prompt_tokens = chunk.usage.prompt_tokens
                 if chunk.usage.completion_tokens:
                     completion_tokens = chunk.usage.completion_tokens # 累加或取最后值

            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_reply += content
                # logger.info(f"收到 chunk: {content}") # 可以取消注释以查看每个 chunk

        # 流结束后记录最终 token 信息
        # 注意：需要根据实际API返回情况调整 token 记录逻辑
        # 这里假设 completion_tokens 在流结束时是准确的
        total_tokens = prompt_tokens + completion_tokens # 简化计算
        logger.info(f"流式响应接收完毕.")
        logger.info(f"输入token数 (估算): {prompt_tokens}")
        logger.info(f"输出token数 (估算): {completion_tokens}")
        logger.info(f"总token数 (估算): {total_tokens}")


        elapsed_time = time.time() - start_time
        logger.info(f"API请求耗时: {elapsed_time:.2f}秒")
        logger.info(f"模型回复: {full_reply}")

        return full_reply
    except Exception as e:
        logger.error(f"调用通义千问大模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 检查是否是 BadRequestError 并提取更具体的 API 错误信息
        if isinstance(e, OpenAI.BadRequestError):
             logger.error(f"API 返回错误详情: {e.body}")
             return f"模型调用失败: {e.body.get('error', {}).get('message', '未知API错误')}"
        return "抱歉，模型调用出错，请稍后再试"

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/client")
async def get_client():
    """返回前端页面HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebRTC语音识别</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; min-height: 100px; }
            button { padding: 10px 20px; margin: 5px; }
            
            /* 添加处理状态指示器样式 */
            .processing-message {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                display: flex;
                align-items: center;
            }
            
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-left: 10px;
                border: 3px solid rgba(0, 0, 0, 0.1);
                border-radius: 50%;
                border-top-color: #007bff;
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            /* 添加用户和AI消息样式 */
            .user-message {
                background-color: #e6f7ff;
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
                text-align: right;
            }
            
            .bot-message {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 10px;
                margin: 5px 0;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <h1>WebRTC语音识别与AI对话</h1>
        <div>
            <button id="startButton">开始录音</button>
            <button id="stopButton" disabled>停止录音</button>
        </div>
        <div id="result">
            <p>识别结果将显示在这里...</p>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let socket;
            let accumulatedChunks = [];
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;
            
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const resultDiv = document.getElementById('result');
            
            function connectWebSocket() {
                socket = new WebSocket(`ws://${window.location.host}/ws`);
                
                socket.onopen = () => {
                    console.log("WebSocket连接已打开");
                    reconnectAttempts = 0;
                    
                    // 添加心跳机制，每15秒发送一次ping保持连接
                    const pingInterval = setInterval(() => {
                        if (socket.readyState === WebSocket.OPEN) {
                            console.log("发送ping保持连接...");
                            socket.send(JSON.stringify({type: "ping"}));
                        } else {
                            clearInterval(pingInterval);
                        }
                    }, 15000);
                };
                
                socket.onclose = (event) => {
                    console.log("WebSocket连接已关闭", event);
                    
                    // 尝试重连
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        console.log(`尝试重连 (${reconnectAttempts}/${maxReconnectAttempts})...`);
                        setTimeout(connectWebSocket, 2000);
                    }
                };
                
                socket.onerror = (error) => {
                    console.error("WebSocket错误:", error);
                };
                
                socket.onmessage = (event) => {
                    console.log("收到服务器消息:", event.data);
                    
                    // 如果是pong消息，不显示在结果区域
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === "pong") {
                            console.log("收到pong响应");
                            return;
                        }
                        
                        // 如果是消息类型
                        if (data.type === "message") {
                            // 移除"正在处理"的提示
                            const processingElements = document.querySelectorAll('.processing-message');
                            processingElements.forEach(el => el.remove());
                            
                            // 根据角色添加不同样式的消息
                            const messageElement = document.createElement('div');
                            messageElement.className = data.role === 'user' ? 'user-message' : 'bot-message';
                            messageElement.textContent = data.content;
                            resultDiv.appendChild(messageElement);
                            
                            // 滚动到底部
                            resultDiv.scrollTop = resultDiv.scrollHeight;
                            return;
                        }
                    } catch (e) {
                        // 不是JSON格式，当作普通文本处理
                    }
                    
                    // 移除"正在处理"的提示
                    const processingElements = document.querySelectorAll('.processing-message');
                    processingElements.forEach(el => el.remove());
                    
                    // 添加识别结果
                    resultDiv.innerHTML += `<p>${event.data}</p>`;
                };
            }
            
            startButton.onclick = async () => {
                // 连接WebSocket
                connectWebSocket();
                
                try {
                    // 获取麦克风权限
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            channelCount: 1,
                            sampleRate: 16000,
                            echoCancellation: true,
                            noiseSuppression: true
                        } 
                    });
                    console.log("麦克风权限已获取");
                    
                    // 使用MediaRecorder API，指定MIME类型
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm;codecs=opus'
                    });
                    
                    // 添加音频可视化，帮助调试
                    const audioContext = new AudioContext();
                    const analyser = audioContext.createAnalyser();
                    const source = audioContext.createMediaStreamSource(stream);
                    source.connect(analyser);
                    
                    // 显示音频电平
                    const audioLevel = document.createElement("div");
                    audioLevel.style.height = "20px";
                    audioLevel.style.width = "0%";
                    audioLevel.style.backgroundColor = "green";
                    audioLevel.style.marginTop = "10px";
                    resultDiv.parentNode.insertBefore(audioLevel, resultDiv);
                    
                    // 更新音频电平显示
                    const dataArray = new Uint8Array(analyser.frequencyBinCount);
                    function updateAudioLevel() {
                        analyser.getByteFrequencyData(dataArray);
                        let sum = 0;
                        for(let i = 0; i < dataArray.length; i++) {
                            sum += dataArray[i];
                        }
                        const average = sum / dataArray.length;
                        const level = Math.min(100, average * 3); // 放大显示
                        audioLevel.style.width = level + "%";
                        audioLevel.style.backgroundColor = level > 5 ? "green" : "red";
                        requestAnimationFrame(updateAudioLevel);
                    }
                    updateAudioLevel();
                    
                    // 收集音频数据
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                        }
                    };
                    
                    // 处理录制完成的音频
                    mediaRecorder.onstop = async () => {
                        console.log("录音停止，处理数据...");
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                        audioChunks = [];
                        
                        if (socket && socket.readyState === WebSocket.OPEN) {
                            console.log(`发送音频数据，大小: ${audioBlob.size} 字节`);
                            const arrayBuffer = await audioBlob.arrayBuffer();
                            socket.send(arrayBuffer);
                            
                            // 添加处理状态指示器
                            const processingMessage = document.createElement('p');
                            processingMessage.className = 'processing-message';
                            processingMessage.innerHTML = '正在处理语音，请稍候... <span class="spinner"></span>';
                            resultDiv.appendChild(processingMessage);
                        }
                        
                        // 继续录音
                        if (!stopButton.disabled) {
                            mediaRecorder.start(6000); // 每6秒一段
                        }
                    };
                    
                    // 开始录音，每6秒一段
                    mediaRecorder.start(6000);
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    
                } catch (error) {
                    console.error("录音错误:", error);
                    resultDiv.innerHTML += `<p>录音错误: ${error.message}</p>`;
                }
            };
            
            stopButton.onclick = () => {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
                
                // 显示等待提示
                const processingMessage = document.createElement('p');
                processingMessage.className = 'processing-message';
                processingMessage.innerHTML = '正在等待最终识别结果，请勿关闭页面... <span class="spinner"></span>';
                resultDiv.appendChild(processingMessage);
                
                // 不立即关闭WebSocket连接，而是等待所有结果返回
                // 只禁用停止按钮，启用开始按钮
                startButton.disabled = false;
                stopButton.disabled = true;
            };
            
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("连接已打开")
    
    # 创建一个任务队列
    processing_tasks = []
    
    try:
        while True:
            try:
                # 先尝试接收文本消息（可能是ping）
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                logger.info(f"收到文本消息: {message}")
                
                # 如果是ping消息，回复pong
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                        logger.info("回复pong消息")
                except:
                    pass
                    
                continue  # 继续等待下一条消息
            except asyncio.TimeoutError:
                # 如果不是文本消息，尝试接收二进制数据
                try:
                    data = await websocket.receive_bytes()
                    logger.info(f"接收到音频数据，开始处理...")
                    
                    # 创建一个异步任务来处理音频，不阻塞WebSocket连接
                    task = asyncio.create_task(process_and_send_result(websocket, data))
                    processing_tasks.append(task)
                    
                    # 清理已完成的任务
                    processing_tasks = [t for t in processing_tasks if not t.done()]
                    
                except:
                    # 如果没有收到任何消息，继续等待
                    await asyncio.sleep(0.1)
                    continue
    except Exception as e:
        logger.error(f"WebSocket连接关闭: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 等待所有处理任务完成
        if processing_tasks:
            logger.info(f"等待 {len(processing_tasks)} 个处理任务完成...")
            await asyncio.gather(*processing_tasks, return_exceptions=True)

async def process_and_send_result(websocket, audio_data):
    """异步处理音频并发送结果"""
    try:
        # 使用Whisper识别语音
        text = await process_audio_with_whisper(audio_data)

        if not text:
            # 如果识别结果为空，不发送给大模型，直接告知用户
            logger.warning("语音识别结果为空，不调用大模型。")
            user_message = {
                "type": "message",
                "role": "user",
                "content": "(未能识别语音)" # 或者保持原来的 text = "未能识别语音，请重试"
            }
            await websocket.send_text(json.dumps(user_message))
            bot_message = {
                "type": "message",
                "role": "assistant",
                "content": "抱歉，我没有听清您说什么，请再说一遍。"
            }
            await websocket.send_text(json.dumps(bot_message))
            return

        # 检查WebSocket是否仍然连接
        try:
            logger.info("准备发送识别结果到客户端...")

            # 发送语音识别结果
            user_message = {
                "type": "message",
                "role": "user",
                "content": text
            }
            await websocket.send_text(json.dumps(user_message))
            logger.info(f"已发送识别结果到客户端: '{text}'")

            # 调用通义千问大模型 (现在是流式处理)
            bot_response = await call_tongyi_model(text)

            # 发送AI回复
            bot_message = {
                "type": "message",
                "role": "assistant",
                "content": bot_response
            }
            await websocket.send_text(json.dumps(bot_message))
            logger.info(f"已发送AI回复到客户端")

        except Exception as e:
            logger.error(f"发送结果失败: {e}")
            # 可以在这里向客户端发送错误消息
            error_message = {
                "type": "message",
                "role": "assistant",
                "content": "抱歉，处理您的请求时发生错误。"
            }
            try:
                await websocket.send_text(json.dumps(error_message))
            except Exception as send_error:
                logger.error(f"向客户端发送错误消息失败: {send_error}")

    except Exception as e:
        logger.error(f"处理音频失败: {e}")
        # 可以在这里向客户端发送错误消息
        error_message = {
            "type": "message",
            "role": "assistant",
            "content": "抱歉，处理音频时发生错误。"
        }
        try:
            await websocket.send_text(json.dumps(error_message))
        except Exception as send_error:
            logger.error(f"向客户端发送错误消息失败: {send_error}")


async def process_audio_with_whisper(audio_data):
    """
    使用Whisper处理音频数据并返回识别的文本
    
    参数:
        audio_data (bytes): 从WebSocket接收的音频数据
        
    返回:
        str: 识别的文本
    """
    try:
        logger.info(f"接收到音频数据，大小: {len(audio_data)} 字节")
        
        # 创建临时文件来保存音频数据
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
            # 直接写入接收到的二进制数据
            temp_audio.write(audio_data)
            
            logger.info(f"音频保存到临时文件: {temp_audio_path}")
            
            # 保存一份音频文件用于调试
            debug_path = os.path.join(os.getcwd(), "debug_audio.webm")
            with open(debug_path, "wb") as f:
                f.write(audio_data)
            logger.info(f"调试音频保存到: {debug_path}")
        
        # 使用线程池执行Whisper处理（避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        # 指定语言为中文
        result = await loop.run_in_executor(None, lambda: model.transcribe(
            temp_audio_path, 
            language="zh"  # 明确指定语言为中文
        ))
        
        # 删除临时文件
        os.unlink(temp_audio_path)
        
        # 返回识别的文本
        text = result["text"].strip()
        logger.info(f"识别结果: '{text}'")
        return text
    except Exception as e:
        logger.error(f"语音识别错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)