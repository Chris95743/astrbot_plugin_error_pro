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
from astrbot.core.provider.entities import ProviderType

@register("astrbot_plugin_error_pro", "Chris", "屏蔽机器人的错误消息，选择是否发送给管理员，支持AI上下文感知的友好解释。", "1.2.0")
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
        self.ai_prompt = self.config.get('ai_prompt', '用户{user_name}在{platform}的{chat_type}中说了："{user_message}"，但是出现了错误：{error}。请用亲切友好的语言先回答用户说的话，再简单的向用户解释出现了什么错误。称呼用户为主人，不要输出你的心理活动，同一报错的解释不要重复。')
        self.ai_timeout = self.config.get('ai_timeout', 10)
        self.ai_max_tokens = self.config.get('ai_max_tokens', 500)
        # 是否屏蔽AI生成的错误解释并发送给管理员
        self.block_ai_explanation_and_send_admin = self.config.get('block_ai_explanation_and_send_admin', False)

        # 关键词触发临时切换 Provider 配置
        self.switch_on_keyword_enable: bool = self.config.get('switch_on_keyword_enable', False)
        # 逗号分隔的关键词字符串；运行时会拆分为列表
        self.switch_keywords: str = self.config.get('switch_keywords', '')
        self.switch_provider_id: str = self.config.get('switch_provider_id', '')
        # 秒；-1 表示不切回
        self.switch_revert_seconds: int = self.config.get('switch_revert_seconds', -1)
        # 触发时是否拦截当次消息
        self.switch_block_message: bool = self.config.get('switch_block_message', True)
        # 触发切换后，是否立即用新 Provider 重试用户原始提问并回复
        self.switch_retry_reply_enable: bool = self.config.get('switch_retry_reply_enable', True)

        # 内部状态：会话级定时切回任务和记录
        self._revert_tasks: dict[str, asyncio.Task] = {}
        self._switch_records: dict[str, tuple[str, str]] = {}

    async def _get_ai_explanation(self, error_message: str, event: AstrMessageEvent = None) -> str:
        """使用AI生成友好的错误解释"""
        if not self.enable_ai_explanation or not self.ai_api_key:
            return None
            
        try:
            # 构建变量字典
            variables = {
                'error': error_message,
                'user_message': '',
                'user_name': '未知用户',
                'platform': '未知平台',
                'chat_type': '未知'
            }
            
            # 如果有event对象，提取用户信息
            if event:
                try:
                    variables['user_message'] = event.get_message_str() or ''
                    variables['user_name'] = event.get_sender_name() or '未知用户'
                    variables['platform'] = event.get_platform_name() or '未知平台'
                    
                    # 判断聊天类型
                    if event.message_obj and event.message_obj.group_id:
                        variables['chat_type'] = '群聊'
                    else:
                        variables['chat_type'] = '私聊'
                        
                except Exception as e:
                    logger.warning(f"获取用户信息失败: {e}")
            
            # 构建请求数据
            prompt = self.ai_prompt.format(**variables)
            
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
        if not message_str:
            try:
                reply_text = ""
                chain = getattr(result, 'chain', None)
                if chain:
                    if isinstance(chain, str):
                        reply_text = chain
                    elif hasattr(chain, '__iter__'):
                        for comp in chain:
                            if hasattr(comp, 'text'):
                                reply_text += str(comp.text)
                            elif isinstance(comp, str):
                                reply_text += comp
                message_str = reply_text
            except Exception as e:
                logger.warning(f"解析回复链失败: {e}")

        if message_str:  # 确保message_str不为空
            # 关键词触发：临时切换 Provider（会话级）
            if await self._maybe_switch_provider_on_keyword(event, message_str):
                return

            # 错误关键词检测
            error_keywords = ['请求失败', '错误类型', '错误信息', '调用失败', '处理失败', '描述失败', '获取模型列表失败']
            if any(keyword in message_str for keyword in error_keywords):
                # 尝试AI解释错误（如果启用）
                ai_explanation = None
                if self.enable_ai_explanation:
                    ai_explanation = await self._get_ai_explanation(message_str, event)
                
                # 如果启用“屏蔽AI解释并发送给管理员”，且AI解释成功，则优先走此分支
                if ai_explanation and self.block_ai_explanation_and_send_admin:
                    await self._send_error_to_admin(event, message_str, ai_explanation=ai_explanation)
                    if self.block_error_messages:
                        logger.info(f"拦截错误消息并屏蔽AI解释: {message_str}")
                        event.stop_event()
                        event.set_result(None)
                    return

                # 否则按原有逻辑：根据开关发送错误信息给管理员
                if self.notify_admin:
                    await self._send_error_to_admin(event, message_str)
                
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
    
    async def _send_error_to_admin(self, event: AstrMessageEvent, message_str: str, ai_explanation: str = None):
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
                    # 组装消息并附加AI解释（如果有）
                    base_message = (
                        f"主人，我在群聊 {group_name}（{chat_id}） 中和 [{user_name}] 聊天出现错误了: {message_str}"
                        if chat_type == "群聊"
                        else f"主人，我在和 {user_name}（{chat_id}） 私聊时出现错误了: {message_str}"
                    )
                    if ai_explanation:
                        base_message = f"{base_message}\nAI解释：{ai_explanation}"
                    await event.bot.send_private_msg(
                        user_id=int(admin_id),
                        message=base_message
                    )
        except Exception as e:
             logger.error(f"Error while sending message to admin: {e}")

    async def _maybe_switch_provider_on_keyword(self, event: AstrMessageEvent, message_str: str) -> bool:
        """检测关键词并按会话临时切换 Provider；根据配置决定是否拦截消息。

        Returns:
            bool: 若已处理并拦截(或无需继续后续流程)返回 True，否则 False。
        """
        try:
            if not self.switch_on_keyword_enable:
                return False

            # 提取关键词（支持中英文逗号、分号、空白分隔）。为空则回退到常见错误关键词
            try:
                import re as _re
                raw = self.switch_keywords or ''
                keywords = [k.strip() for k in _re.split(r"[，,;；\s]+", raw) if k and k.strip()]
            except Exception:
                keywords = [k.strip() for k in (self.switch_keywords or '').split(',') if k and k.strip()]
            if not keywords:
                keywords = ['请求失败', '错误类型', '错误信息', '调用失败', '处理失败', '描述失败', '获取模型列表失败']

            # 忽略大小写匹配
            msg_l = (message_str or '').lower()
            if not any((kw or '').lower() in msg_l for kw in keywords):
                logger.debug("关键词切换未触发：未匹配到关键词。")
                return False

            if not self.switch_provider_id:
                logger.warning("检测到关键词，但未配置 switch_provider_id，已跳过临时切换。")
                return self.switch_block_message and self._intercept(event)

            # 会话隔离开关（未开启则回退到全局切换）
            provider_settings = self.context.get_config().get("provider_settings", {})
            separate = provider_settings.get("separate_provider", False)

            # 目标提供商校验
            target = self.context.get_provider_by_id(self.switch_provider_id)
            if not target:
                logger.error(f"未找到目标提供商: {self.switch_provider_id}")
                return self.switch_block_message and self._intercept(event)

            umo = event.unified_msg_origin
            prev = self.context.get_using_provider(umo=umo)
            prev_id = prev.meta().id if prev else None

            if prev_id == self.switch_provider_id:
                logger.info(f"会话 {umo} 已在使用 {self.switch_provider_id}，无需切换。")
                return self.switch_block_message and self._intercept(event)

            # 执行切换（优先会话级，未开启会话隔离则进行全局切换）
            if separate:
                await self.context.provider_manager.set_provider(
                    provider_id=self.switch_provider_id,
                    provider_type=ProviderType.CHAT_COMPLETION,
                    umo=umo,
                )
                logger.info(f"已将会话 {umo} 的 Provider 切换为 {self.switch_provider_id}。")
            else:
                await self.context.provider_manager.set_provider(
                    provider_id=self.switch_provider_id,
                    provider_type=ProviderType.CHAT_COMPLETION,
                )
                logger.info(f"provider_settings.separate_provider 未开启，已执行全局 Provider 切换为 {self.switch_provider_id}。")

            # 取消已有回退任务
            if umo in self._revert_tasks:
                try:
                    self._revert_tasks[umo].cancel()
                except Exception:
                    pass
                finally:
                    self._revert_tasks.pop(umo, None)

            self._switch_records[umo] = (self.switch_provider_id, prev_id or "")

            # 定时切回
            if isinstance(self.switch_revert_seconds, int) and self.switch_revert_seconds > 0 and prev_id:
                async def _revert():
                    try:
                        await asyncio.sleep(self.switch_revert_seconds)
                        # 仅当当前仍为目标 Provider 时才回退
                        curr = self.context.get_using_provider(umo=umo)
                        if curr and curr.meta().id == self.switch_provider_id:
                            await self.context.provider_manager.set_provider(
                                provider_id=prev_id,
                                provider_type=ProviderType.CHAT_COMPLETION,
                                umo=umo if separate else None,
                            )
                            logger.info(f"已将会话 {umo} 的 Provider 从 {self.switch_provider_id} 回退到 {prev_id}。")
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"回退 Provider 失败: {e}")
                    finally:
                        self._revert_tasks.pop(umo, None)
                        self._switch_records.pop(umo, None)

                self._revert_tasks[umo] = asyncio.create_task(_revert(), name=f"provider_revert_{umo}")
            elif self.switch_revert_seconds == -1:
                logger.info(f"会话 {umo} 配置为不回退 Provider（-1）。")

            # 触发后尝试立即用新 Provider 回复原始提问
            if self.switch_retry_reply_enable:
                try:
                    success = await self._reply_with_switched_provider(event)
                    if success:
                        return True
                except Exception as e:
                    logger.error(f"切换后自动重试回复失败: {e}")

            # 如果未启用自动回复，则根据配置决定是否拦截当次消息
            return self.switch_block_message and self._intercept(event)

        except Exception as e:
            logger.error(f"处理关键词临时切换 Provider 失败: {e}")
            return False

    def _intercept(self, event: AstrMessageEvent) -> bool:
        """拦截当前消息并清空结果。"""
        try:
            event.stop_event()
            event.set_result(None)
        except Exception:
            pass
        return True

    async def _reply_with_switched_provider(self, event: AstrMessageEvent) -> bool:
        """在完成 Provider 切换后，使用新 Provider 重试用户原始提问并直接回复。

        Returns:
            bool: 成功设定了新的回复则返回 True。
        """
        try:
            prov = self.context.get_using_provider(event.unified_msg_origin)
            if not prov:
                logger.error("切换后未获取到 Provider，无法自动回复。")
                return False

            # 准备对话上下文
            conv_mgr = self.context.conversation_manager
            cid = await conv_mgr.get_curr_conversation_id(event.unified_msg_origin)
            if not cid:
                cid = await conv_mgr.new_conversation(event.unified_msg_origin)
            conversation = await conv_mgr.get_conversation(event.unified_msg_origin, cid)

            # 构建 ProviderRequest（使用与核心流程一致的方法）
            req = event.request_llm(
                prompt=event.get_message_str(),
                func_tool_manager=self.context.get_llm_tool_manager(),
                conversation=conversation,
            )
            # 填充上下文
            try:
                req.contexts = json.loads(conversation.history or "[]")
            except Exception:
                req.contexts = []

            # 调用新 Provider 进行一次非流式回复
            llm_resp = await prov.text_chat(**req.__dict__)
            if not llm_resp:
                return False

            # 将回复写入结果并终止事件传播
            if llm_resp.result_chain and getattr(llm_resp.result_chain, 'chain', None):
                event.set_result(MessageEventResult(chain=llm_resp.result_chain.chain))
            else:
                event.set_result(MessageEventResult().message(llm_resp.completion_text or ""))
            event.stop_event()
            return True
        except Exception as e:
            logger.error(f"自动重试回复异常: {e}")
            return False
