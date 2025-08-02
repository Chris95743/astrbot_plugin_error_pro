import re
import aiohttp
import asyncio
import json
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.platform import AstrBotMessage
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.provider import LLMResponse
from openai.types.chat.chat_completion import ChatCompletion
import traceback

@register("error_pro", "Chris", "屏蔽机器人的错误信息回复，发送给管理员，支持AI友好解释。", "1.1.0")
class ErrorFilter(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        # 是否屏蔽错误信息（阻止发送给用户）
        self.block_error_messages = self.config.get('block_error_messages', True)
        # 是否将错误信息发送给管理员
        self.notify_admin = self.config.get('notify_admin', True)
        # 管理员列表
        self.admins_id: list = context.get_config().get("admins_id", [])
        # AI相关配置
        self.enable_ai_explanation = self.config.get('enable_ai_explanation', False)
        self.ai_base_url = self.config.get('ai_base_url', 'https://api.openai.com/v1')
        self.ai_api_key = self.config.get('ai_api_key', '')
        self.ai_model = self.config.get('ai_model', 'gpt-3.5-turbo')
        self.ai_prompt = self.config.get('ai_prompt', '请将以下技术错误信息转换为普通用户能理解的友好提示，保持简洁明了，不超过50字：{error}')
        self.ai_timeout = self.config.get('ai_timeout', 10)
        self.ai_max_tokens = self.config.get('ai_max_tokens', 100)

    async def _get_ai_explanation(self, error_message: str) -> str:
        """使用AI生成友好的错误解释"""
        if not self.enable_ai_explanation or not self.ai_api_key:
            return None
            
        try:
            # 构建请求数据
            prompt = self.ai_prompt.format(error=error_message)
            
            headers = {
                'Authorization': f'Bearer {self.ai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.ai_model,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': self.ai_max_tokens,
                'temperature': 0.3
            }
            
            # 调用AI API
            timeout = aiohttp.ClientTimeout(total=self.ai_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.ai_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            ai_explanation = result['choices'][0]['message']['content'].strip()
                            logger.info(f"AI生成的错误解释: {ai_explanation}")
                            return ai_explanation
                    else:
                        logger.error(f"AI API调用失败: {response.status} - {await response.text()}")
                        
        except asyncio.TimeoutError:
            logger.error("AI API调用超时")
        except Exception as e:
            logger.error(f"AI API调用异常: {e}")
            
        return None

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        result = event.get_result()
        if not result:  # 检查结果是否存在
            return

        message_str = result.get_plain_text()
        if message_str:  # 确保message_str不为空
            # 错误关键词检测
            error_keywords = ['请求失败', '错误类型', '错误信息', '调用失败', '处理失败', '描述失败', '获取模型列表失败']
            if any(keyword in message_str for keyword in error_keywords):
                # 发送给管理员（如果开启了通知功能）
                if self.notify_admin:
                    await self._send_error_to_admin(event, message_str)
                
                # 尝试AI解释错误（如果启用）
                ai_explanation = None
                if self.enable_ai_explanation:
                    ai_explanation = await self._get_ai_explanation(message_str)
                
                # 屏蔽原错误消息
                if self.block_error_messages:
                    logger.info(f"拦截错误消息: {message_str}")
                    event.stop_event()  # 停止事件传播
                    
                    # 如果AI解释成功，设置AI解释作为返回结果
                    if ai_explanation:
                        logger.info(f"使用AI解释替换错误消息: {ai_explanation}")
                        event.set_result(event.plain_result(ai_explanation))
                    else:
                        event.set_result(None)  # 清除结果
                return  # 确保后续处理不会执行
    
    async def _send_error_to_admin(self, event: AstrMessageEvent, message_str: str):
        """发送错误信息给管理员"""
        # 获取事件信息
        chat_type = "未知类型"
        chat_id = "未知ID"
        user_name = "未知用户"
        platform_name = "未知平台"
        group_name = "未知群聊"  # 初始化群聊名称

        try:  # Catch potential exceptions during event object processing
            if event.message_obj:
                if event.message_obj.group_id:
                    chat_type = "群聊"
                    chat_id = event.message_obj.group_id

                    # 尝试获取群聊名称
                    try:
                        # 使用 event.bot.get_group_info 获取群组信息
                        group_info = await event.bot.get_group_info(group_id=chat_id)

                        # 假设群信息对象里有 group_name 属性
                        group_name = group_info.get('group_name', "获取群名失败") if group_info else "获取群名失败"

                    except Exception as e:
                        logger.error(f"获取群名失败: {e}")
                        group_name = "获取群名失败"  # 设置为错误提示
                else:
                    chat_type = "私聊"
                    chat_id = event.message_obj.sender.user_id

                user_name = event.get_sender_name()
                platform_name = event.get_platform_name()  # 获取平台名称
            else:
                logger.warning("event.message_obj is None. Could not get chat details")

        except Exception as e:
            logger.error(f"Error while processing event information: {e}")

        # 给管理员发通知
        try: #Catch exceptions while sending message to admin.
            for admin_id in self.admins_id:
                if str(admin_id).isdigit():  # 管理员QQ号
                    # Include group_name in the message
                    await event.bot.send_private_msg(
                        user_id=int(admin_id),
                        message=f"主人，我在群聊 {group_name}（{chat_id}） 中和 [{user_name}] 聊天出现错误了: {message_str}" if chat_type == "群聊" else f"主人，我在和 {user_name}（{chat_id}） 私聊时出现错误了: {message_str}"
                    )
        except Exception as e:
             logger.error(f"Error while sending message to admin: {e}")
