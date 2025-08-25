import re
import aiohttp
import asyncio
import json
import copy
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.core.provider.entities import ProviderType
import astrbot.api.message_components as Comp

class SimpleRetryLogic:
    """简化的重试逻辑，基于astrabot_plugin_retry插件的简化设计（虽然我早就想吐槽这个astrabot打错了）"""
    
    def __init__(self, context: Context, config: dict):
        self.context = context
        self.config = config
        
        # 重试配置
        self.max_attempts = config.get('retry_max_attempts', 2)
        self.retry_delay = config.get('retry_delay', 2)
        
        # 错误关键词配置 - 完全从配置文件获取
        retry_keywords = config.get('retry_error_keywords_list', [])
        self.error_keywords = [str(k).strip().lower() for k in retry_keywords if str(k).strip()]
        
        # 人设配置
        self.always_use_system_prompt = config.get('always_use_system_prompt', True)
        self.fallback_system_prompt = config.get('fallback_system_prompt', '').strip()
        
        # 记录最后一次重试的错误信息
        # 记录最后一次重试的错误信息
        self.last_error_info = ""
        
        logger.info(f"[SimpleRetry] 已初始化，最大重试次数: {self.max_attempts}, 重试延迟: {self.retry_delay}秒")
    

    
    
    async def _get_complete_context(self, unified_msg_origin: str):
        """获取完整的对话上下文"""
        try:
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(unified_msg_origin)
            if not curr_cid:
                return []
            
            conv = await self.context.conversation_manager.get_conversation(unified_msg_origin, curr_cid)
            if not conv or not conv.history:
                return []
            
            context_history = await asyncio.to_thread(json.loads, conv.history)
            return context_history
            
        except Exception as e:
            logger.error(f"[SimpleRetry] 获取对话上下文时发生错误: {e}")
            return []
    
    async def _get_provider_config(self):
        """获取LLM提供商的配置"""
        provider = self.context.get_using_provider()
        if not provider:
            return None, None, None
        
        # 获取系统提示词
        system_prompt = None
        if hasattr(provider, "system_prompt"):
            system_prompt = provider.system_prompt
        elif hasattr(provider, "config") and provider.config:
            system_prompt = provider.config.get("system_prompt")
        
        # 获取工具配置
        func_tool = None
        if hasattr(provider, "func_tool"):
            func_tool = provider.func_tool
        
        return provider, system_prompt, func_tool
    
    async def _perform_retry_with_context(self, event: AstrMessageEvent):
        """执行重试，保持完整上下文和人设"""
        provider, system_prompt, func_tool = await self._get_provider_config()
        
        if not provider:
            logger.warning("[SimpleRetry] LLM提供商未启用，无法重试")
            return None

        try:
            # 获取完整的对话上下文
            context_history = await self._get_complete_context(event.unified_msg_origin)
            
            # 获取图片URL
            image_urls = []
            try:
                image_urls = [
                    comp.url
                    for comp in event.message_obj.message
                    if isinstance(comp, Comp.Image) and hasattr(comp, "url") and comp.url
                ]
            except Exception:
                pass

            logger.debug(f"[SimpleRetry] 正在使用完整上下文进行重试... Prompt: '{event.message_str}'")
            logger.debug(f"[SimpleRetry] 上下文长度: {len(context_history)}")

            # 人设处理
            if self.always_use_system_prompt:
                if (not system_prompt) and self.fallback_system_prompt:
                    system_prompt = self.fallback_system_prompt
                    logger.debug("[SimpleRetry] 使用备用人设")

                if system_prompt:
                    # 移除上下文中的所有system消息
                    filtered = []
                    removed = 0
                    for m in context_history:
                        if isinstance(m, dict) and str(m.get('role', '')).lower() == 'system':
                            removed += 1
                            continue
                        filtered.append(m)
                    if removed > 0:
                        logger.debug(f"[SimpleRetry] 已强制覆盖人设：移除 {removed} 条历史 system 消息")
                    context_history = filtered
            
            # 构建重试请求
            kwargs = {
                'prompt': event.message_str,
                'contexts': context_history,
                'image_urls': image_urls,
                'func_tool': func_tool,
            }
            
            if self.always_use_system_prompt and system_prompt:
                kwargs['system_prompt'] = system_prompt

            llm_response = await provider.text_chat(**kwargs)
            return llm_response
            
        except Exception as e:
            logger.error(f"[SimpleRetry] 重试调用LLM时发生错误: {e}")
            # 保存具体的错误信息
            self.last_error_info = str(e)
            return None
    
    async def execute_retry(self, event: AstrMessageEvent) -> bool:
        """执行重试逻辑，返回是否成功"""
        if self.max_attempts <= 0:
            return False
        
        # 检查是否已经处理过重试，防止重复处理
        if hasattr(event, '_simple_retry_processed'):
            logger.debug("[SimpleRetry] 重试逻辑已处理过，跳过重复处理")
            return False
        
        # 标记已开始处理重试
        event._simple_retry_processed = True
        
        # 检查是否需要重试
        result = event.get_result()
        if not self._should_retry_simple(result):
            return False
        
        # 检查工具调用
        _llm_response = getattr(event, 'llm_response', None)
        if _llm_response and hasattr(_llm_response, 'choices') and _llm_response.choices:
            finish_reason = getattr(_llm_response.choices[0], 'finish_reason', None)
            if finish_reason == 'tool_calls':
                logger.debug("[SimpleRetry] 检测到正常的工具调用，不进行干预")
                return False
        
        # 检查用户消息
        if not event.message_str or not event.message_str.strip():
            logger.debug("[SimpleRetry] 用户消息为空，跳过重试")
            return False

        logger.info("[SimpleRetry] 检测到需要重试的情况，开始重试流程")

        # 执行重试
        delay = max(0, int(self.retry_delay))
        last_error_info = ""
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"[SimpleRetry] 第 {attempt}/{self.max_attempts} 次重试...")

            new_response = await self._perform_retry_with_context(event)

            if not new_response or not getattr(new_response, 'completion_text', ''):
                logger.warning(f"[SimpleRetry] 第 {attempt} 次重试返回空结果")
                # 如果在_perform_retry_with_context中已经设置了具体错误信息，则使用它
                # 否则使用默认的错误描述
                if not self.last_error_info:
                    last_error_info = "重试返回空结果"
                else:
                    last_error_info = self.last_error_info
                if attempt < self.max_attempts and delay > 0:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                continue

            new_text = new_response.completion_text.strip()

            # 检查新回复质量
            new_text_lower = new_text.lower()
            has_error = any(keyword in new_text_lower for keyword in self.error_keywords)
            
            # 检查是否缺少#结束#标记（如果启用了完成检测）
            missing_completion = (self.config.get('enable_completion_check', False) and 
                                '#结束#' not in new_text)

            if new_text and not has_error and not missing_completion:
                logger.info(f"[SimpleRetry] 第 {attempt} 次重试成功，生成有效回复")
                
                # 如果启用了完成检测，删除#结束#标记
                final_text = new_text
                if self.config.get('enable_completion_check', False) and '#结束#' in final_text:
                    final_text = final_text.replace('#结束#', '').strip()
                    logger.debug("[SimpleRetry] 重试成功后删除了#结束#标记")
                
                # 直接替换结果，保持事件流程的完整性
                event.set_result(event.plain_result(final_text))
                return True
            else:
                if has_error:
                    error_reason = "重试仍包含错误关键词"
                elif missing_completion:
                    error_reason = "重试仍缺少#结束#标记"
                else:
                    error_reason = "重试返回空内容"
                logger.warning(f"[SimpleRetry] 第 {attempt} 次{error_reason}: {new_text[:100]}...")
                last_error_info = error_reason
                if attempt < self.max_attempts and delay > 0:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)

        logger.error(f"[SimpleRetry] 所有 {self.max_attempts} 次重试均失败，最后错误: {last_error_info}")
        # 保存最后一次错误信息，供外部使用
        self.last_error_info = last_error_info
        return False
    
    def _should_retry_simple(self, result) -> bool:
        """判断是否需要重试：只在真正需要重试的情况下才触发"""
        try:
            # 如果没有结果，不需要重试
            if not result:
                return False
            
            # 检查是否有实际内容
            has_content = False
            try:
                if hasattr(result, 'chain') and result.chain:
                    has_content = any(
                        (isinstance(c, Comp.Plain) and c.text.strip())
                        or not isinstance(c, Comp.Plain)
                        for c in result.chain
                    )
                elif hasattr(result, 'get_plain_text'):
                    text = result.get_plain_text()
                    has_content = text and text.strip()
            except Exception:
                pass
            
            # 如果有内容，检查是否包含错误关键词
            if has_content:
                try:
                    text = result.get_plain_text() if hasattr(result, 'get_plain_text') else str(result)
                    if text:
                        text_lower = text.lower()
                        # 使用配置中的错误关键词
                        for keyword in self.error_keywords:
                            if keyword in text_lower:
                                logger.debug(f"[SimpleRetry] 检测到错误关键词 '{keyword}'，需要重试")
                                return True
                        
                        # 检查#结束#标记（如果启用了完成检测）
                        if self.config.get('enable_completion_check', False) and '#结束#' not in text:
                            logger.debug("[SimpleRetry] 未检测到#结束#标记，需要重试")
                            return True
                except Exception:
                    pass
                
                # 有内容且没有错误关键词且有完成标记（或未启用检测），不需要重试
                return False
            
            # 没有内容，需要重试
            logger.debug("[SimpleRetry] 检测到空回复，需要重试")
            return True
            
        except Exception as e:
            logger.error(f"[SimpleRetry] 判断是否需要重试时发生错误: {e}")
            return False

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
        # 是否屏蔽重试失败提示并发送给管理员
        self.block_retry_fail_and_send_admin = self.config.get('block_retry_fail_and_send_admin', False)

        # 初始化简化的重试逻辑（可配置是否启用）
        retry_enable = config.get('retry_enable', False)
        if retry_enable:
            self.simple_retry = SimpleRetryLogic(context, config)
            logger.info("[ErrorPro] 重试功能已启用")
        else:
            self.simple_retry = None
            logger.info("[ErrorPro] 重试功能已禁用")

        # 重试失败后自动切换 Provider 配置
        self.auto_switch_on_retry_fail: bool = self.config.get('auto_switch_on_retry_fail', False)
        self.switch_provider_id: str = self.config.get('switch_provider_id', '')
        # 秒；-1 表示不切回
        self.switch_revert_seconds: int = self.config.get('switch_revert_seconds', -1)
        # 切换后是否立即用新 Provider 重试用户原始提问并回复
        self.switch_retry_reply_enable: bool = self.config.get('switch_retry_reply_enable', True)
        # 切换后是否通知管理员
        self.switch_notify_admin: bool = bool(self.config.get('switch_notify_admin', False))
        # 是否屏蔽重试错误信息并发送给管理员
        self.block_retry_error_and_send_admin: bool = self.config.get('block_retry_error_and_send_admin', False)
        
        # 失败兜底提示（仅在未启用自动切换时使用）
        self.retry_fail_message: str = str(self.config.get('retry_fail_message', '抱歉，我已尝试重试但仍未成功，请稍后再试。')).strip()

        # #结束#标记检测配置
        self.enable_completion_check: bool = self.config.get('enable_completion_check', False)
        self.completion_check_retry_enable: bool = self.config.get('completion_check_retry_enable', True)
        
        # 人设注入配置
        self.inject_completion_prompt: bool = self.config.get('inject_completion_prompt', False)
        self.completion_prompt_text: str = self.config.get('completion_prompt_text', '你的回复结束必须在末尾加上#结束#，否则系统将会判定你的回复无效。这个要求十分重要!!!')
        
        # 人设备份和注入
        self.persona_backup = None
        if self.inject_completion_prompt:
            self._inject_completion_prompt()

        # 内部状态：会话级定时切回任务和记录
        self._revert_tasks: dict[str, asyncio.Task] = {}
        self._switch_records: dict[str, tuple[str, str]] = {}
        
        # 用于跟踪#结束#标记检测状态
        self._completion_marks: dict[str, bool] = {}

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

    def _check_and_remove_completion_mark(self, text: str) -> tuple[bool, str]:
        """检查#结束#标记并删除
        
        Returns:
            tuple[bool, str]: (是否包含结束标记, 删除标记后的文本)
        """
        if not text:
            return False, text
            
        # 检查是否包含#结束#标记
        has_completion_mark = '#结束#' in text
        
        # 删除#结束#标记
        clean_text = text.replace('#结束#', '').strip()
        
        return has_completion_mark, clean_text

    def _inject_completion_prompt(self):
        """注入#结束#标记提示到AI人设中"""
        try:
            personas = self.context.provider_manager.personas
            if not personas:
                logger.warning("[ErrorPro] 没有找到可用的人设，跳过注入")
                return
            
            # 备份原始人设
            self.persona_backup = copy.deepcopy(personas)
            
            # 为每个人设注入提示
            for persona in personas:
                if "prompt" in persona:
                    # 检查是否已经包含#结束#相关提示，避免重复注入
                    if "#结束#" not in persona["prompt"]:
                        persona["prompt"] = persona["prompt"] + "\n\n" + self.completion_prompt_text
                        logger.info("[ErrorPro] 已向人设注入#结束#标记提示")
                    else:
                        logger.debug("[ErrorPro] 人设中已包含#结束#标记提示，跳过注入")
                        
        except Exception as e:
            logger.error(f"[ErrorPro] 人设注入失败: {e}")

    def _restore_persona(self):
        """恢复原始人设"""
        try:
            if self.persona_backup:
                personas = self.context.provider_manager.personas
                for persona, persona_backup in zip(personas, self.persona_backup):
                    persona["prompt"] = persona_backup["prompt"]
                logger.info("[ErrorPro] 已恢复原始人设")
        except Exception as e:
            logger.error(f"[ErrorPro] 人设恢复失败: {e}")

    @filter.on_llm_response(priority=95)
    async def on_llm_response(self, event: AstrMessageEvent, response):
        """在LLM响应阶段处理#结束#标记"""
        if not self.enable_completion_check:
            return
            
        if not response or not hasattr(response, 'completion_text') or not response.completion_text:
            return
            
        text = response.completion_text
        event_id = id(event)  # 使用事件ID作为唯一标识
        
        if '#结束#' in text:
            # 记录检测到了#结束#标记
            self._completion_marks[event_id] = True
            # 删除#结束#标记
            clean_text = text.replace('#结束#', '').strip()
            response.completion_text = clean_text
            logger.debug("[ErrorPro] 在LLM响应阶段检测到并删除了#结束#标记")
        else:
            # 记录没有检测到#结束#标记
            self._completion_marks[event_id] = False
            logger.debug("[ErrorPro] 在LLM响应阶段未检测到#结束#标记")

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        # 检查是否已经处理过，防止重复处理
        if hasattr(event, '_errorpro_main_processed'):
            logger.debug("[ErrorPro] 主处理逻辑已处理过，跳过重复处理")
            return
        
        # 标记已开始处理
        event._errorpro_main_processed = True
        
        result = event.get_result()
        message_str = None

        if not result:  # 检查结果是否存在
            message_str = None
        else:
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

        # 使用简化的重试逻辑
        retry_success = False
        
        # 检查是否已经处理过重试，防止重复处理
        if hasattr(event, '_errorpro_retry_processed'):
            logger.debug("[ErrorPro] 重试逻辑已处理过，跳过重复处理")
            return
        
        # 只有在启用了重试功能的情况下才执行重试逻辑
        need_retry = False
        event_id = id(event)
        
        # 检查是否启用了重试功能
        retry_enabled = self.config.get('retry_enable', False)
        if not retry_enabled:
            logger.debug("[ErrorPro] 重试功能已禁用，跳过错误关键词检测")
        elif result and hasattr(result, 'get_plain_text'):
            text = result.get_plain_text()
            # 使用配置中的错误关键词
            error_keywords = self.simple_retry.error_keywords if self.simple_retry else []
            if text and any(keyword in text.lower() for keyword in error_keywords):
                need_retry = True
                logger.debug("[ErrorPro] 检测到错误关键词，需要重试")
        
        # 检查#结束#标记状态（如果启用了完成检测和重试功能）
        completion_mark_missing = False
        retry_enabled = self.config.get('retry_enable', False)
        if retry_enabled and self.enable_completion_check and event_id in self._completion_marks:
            if not self._completion_marks[event_id]:
                need_retry = True
                completion_mark_missing = True
                logger.debug("[ErrorPro] 未检测到#结束#标记，需要重试")
            # 清理标记记录
            self._completion_marks.pop(event_id, None)
        elif not retry_enabled and self.enable_completion_check and event_id in self._completion_marks:
            # 如果重试功能已禁用，仍然需要清理标记记录
            self._completion_marks.pop(event_id, None)
            logger.debug("[ErrorPro] 重试功能已禁用，跳过#结束#标记检测")
        
        # 执行重试逻辑
        retry_success = False
        
        # 根据类型决定是否进行重试
        should_attempt_retry = False
        if need_retry:
            if completion_mark_missing:
                # 对于#结束#标记缺失，检查专门的重试开关
                should_attempt_retry = self.completion_check_retry_enable
                logger.debug(f"[ErrorPro] #结束#标记缺失，重试开关状态: {should_attempt_retry}")
                if not should_attempt_retry:
                    logger.info("[ErrorPro] 检测到缺少#结束#标记，但重试功能已禁用，直接进入后续处理")
            else:
                # 对于其他错误，只有在SimpleRetry启用时才重试
                should_attempt_retry = self.simple_retry is not None
                logger.debug(f"[ErrorPro] 错误关键词检测，SimpleRetry状态: {self.simple_retry is not None}")
        
        logger.debug(f"[ErrorPro] 重试决策 - need_retry: {need_retry}, should_attempt_retry: {should_attempt_retry}, simple_retry: {self.simple_retry is not None}")
        
        # 执行重试（如果需要且可以）
        if should_attempt_retry:
            if self.simple_retry:
                try:
                    logger.info("[ErrorPro] 开始执行重试逻辑")
                    retry_success = await self.simple_retry.execute_retry(event)
                    if retry_success:
                        # 标记已处理重试
                        event._errorpro_retry_processed = True
                        return
                except Exception as e:
                    logger.error(f"[ErrorPro] 简化重试执行异常: {e}")
            else:
                logger.warning("[ErrorPro] 需要重试但SimpleRetry未初始化，跳过重试直接进入后续处理")
        
        # 重试失败后的处理
        if need_retry and not retry_success:
            # 如果启用了自动切换Provider
            if self.auto_switch_on_retry_fail and self.switch_provider_id:
                logger.info("[ErrorPro] 重试失败，尝试自动切换Provider")
                if await self._auto_switch_provider_on_retry_fail(event):
                    return
            # 如果未启用自动切换，使用失败提示
            elif self.retry_fail_message:
                logger.info("[ErrorPro] 重试失败，进入失败提示处理")
                # 获取最后一次重试的错误信息
                last_error_info = ""
                if hasattr(self.simple_retry, "last_error_info"):
                    last_error_info = getattr(self.simple_retry, "last_error_info", "")
                
                # 将错误信息添加到失败提示中
                fail_message = self.retry_fail_message
                
                # 根据配置决定是否在用户提示中显示错误信息
                if self.block_retry_error_and_send_admin and last_error_info:
                    # 屏蔽错误信息，使用模板中的变量占位符
                    fail_message = fail_message.replace("{last_error_info}", "")
                    # 发送错误信息给管理员
                    if self.admins_id:
                        admin_message = f"[ErrorPro] 用户 {event.get_user_id()} 的请求重试失败，错误信息：{last_error_info}"
                        for admin_id in self.admins_id:
                            try:
                                asyncio.create_task(self.context.post_message(admin_message, admin_id))
                                logger.info(f"[ErrorPro] 已将重试错误信息发送给管理员 {admin_id}")
                            except Exception as e:
                                logger.error(f"[ErrorPro] 发送重试错误信息给管理员 {admin_id} 失败: {e}")
                else:
                    # 直接替换模板中的变量
                    fail_message = fail_message.replace("{last_error_info}", last_error_info)
                
                # 若开启“屏蔽重试失败提示并发送给管理员”，则不向用户发送，转而仅通知管理员
                if self.block_retry_fail_and_send_admin:
                    try:
                        await self._send_error_to_admin(event, fail_message)
                    except Exception as e:
                        logger.error(f"[ErrorPro] 发送重试失败提示给管理员时出错: {e}")
                    # 标记已处理重试并阻止向用户发送
                    event._errorpro_retry_processed = True
                    event.set_result(None)
                    event.stop_event()
                    return
                
                # 默认行为：将失败提示发送给用户
                event._errorpro_retry_processed = True
                event.set_result(event.plain_result(fail_message))
                event.stop_event()
                return

        # 若有文本，进行错误检测和处理
        if message_str:


            # 错误关键词检测
            # 检查是否启用了重试功能
            retry_enabled = self.config.get('retry_enable', False)
            if retry_enabled:
                error_keywords = self.simple_retry.error_keywords if self.simple_retry else self.config.get('retry_error_keywords_list', [])
                has_error_keywords = any(keyword in message_str.lower() for keyword in error_keywords)
            else:
                # 如果重试功能已禁用，仍然需要检测错误关键词，但使用默认列表
                default_error_keywords = ['请求失败', '错误类型', '错误信息', '调用失败', '处理失败', '描述失败', '获取模型列表失败']
                has_error_keywords = any(keyword in message_str for keyword in default_error_keywords)
                
            if has_error_keywords:
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
                
                # 清理完成标记记录
                event_id = id(event)
                self._completion_marks.pop(event_id, None)
                return  # 确保后续处理不会执行
        
        # 最后清理完成标记记录（如果没有触发错误处理）
        event_id = id(event)
        self._completion_marks.pop(event_id, None)

    async def terminate(self):
        """插件销毁时的清理工作"""
        # 恢复原始人设
        if self.inject_completion_prompt:
            self._restore_persona()
        
        # 清理其他资源
        self._completion_marks.clear()
        self._revert_tasks.clear()
        self._switch_records.clear()
        
        logger.info("[ErrorPro] 插件资源清理完成")

    @filter.command("重新加载完成标记提示")
    async def reload_completion_prompt(self, event: AstrMessageEvent):
        """重新加载#结束#标记提示设置"""
        try:
            # 先恢复原始人设
            if self.persona_backup:
                self._restore_persona()
            
            # 重新读取配置
            self.inject_completion_prompt = self.config.get('inject_completion_prompt', False)
            self.completion_prompt_text = self.config.get('completion_prompt_text', '你的回复结束必须在末尾加上#结束#，否则系统将会判定你的回复无效。这个要求十分重要!!!')
            
            # 如果启用了注入，重新注入
            if self.inject_completion_prompt:
                self._inject_completion_prompt()
                yield event.plain_result("✅ 已重新注入#结束#标记提示到AI人设中")
            else:
                yield event.plain_result("✅ 已恢复原始AI人设，#结束#标记提示已移除")
                
        except Exception as e:
            logger.error(f"[ErrorPro] 重新加载完成标记提示失败: {e}")
            yield event.plain_result(f"❌ 重新加载失败: {e}")

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
    
    async def _auto_switch_provider_on_retry_fail(self, event: AstrMessageEvent) -> bool:
        """重试失败后自动切换Provider并重试用户原始提问

        Returns:
            bool: 成功切换并回复则返回 True。
        """
        try:
            if not self.switch_provider_id:
                logger.warning("[ErrorPro] 未配置 switch_provider_id，无法自动切换")
                return False

            # 会话隔离开关（未开启则回退到全局切换）
            provider_settings = self.context.get_config().get("provider_settings", {})
            separate = provider_settings.get("separate_provider", False)

            # 目标提供商校验
            target = self.context.get_provider_by_id(self.switch_provider_id)
            if not target:
                logger.error(f"[ErrorPro] 未找到目标提供商: {self.switch_provider_id}")
                return False

            umo = event.unified_msg_origin
            prev = self.context.get_using_provider(umo=umo)
            prev_id = prev.meta().id if prev else None

            if prev_id == self.switch_provider_id:
                logger.info(f"[ErrorPro] 会话 {umo} 已在使用 {self.switch_provider_id}，无需切换")
                return False

            # 执行切换（优先会话级，未开启会话隔离则进行全局切换）
            if separate:
                await self.context.provider_manager.set_provider(
                    provider_id=self.switch_provider_id,
                    provider_type=ProviderType.CHAT_COMPLETION,
                    umo=umo,
                )
                logger.info(f"[ErrorPro] 已将会话 {umo} 的 Provider 切换为 {self.switch_provider_id}")
            else:
                await self.context.provider_manager.set_provider(
                    provider_id=self.switch_provider_id,
                    provider_type=ProviderType.CHAT_COMPLETION,
                )
                logger.info(f"[ErrorPro] provider_settings.separate_provider 未开启，已执行全局 Provider 切换为 {self.switch_provider_id}")

            # 切换完成后可选通知管理员
            if self.switch_notify_admin:
                try:
                    info_msg = f"重试失败后自动切换Provider为: {self.switch_provider_id}"
                    await self._send_error_to_admin(event, info_msg)
                except Exception as _:
                    pass

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
                            logger.info(f"[ErrorPro] 已将会话 {umo} 的 Provider 从 {self.switch_provider_id} 回退到 {prev_id}")
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"[ErrorPro] 回退 Provider 失败: {e}")
                    finally:
                        self._revert_tasks.pop(umo, None)
                        self._switch_records.pop(umo, None)

                self._revert_tasks[umo] = asyncio.create_task(_revert(), name=f"provider_revert_{umo}")
            elif self.switch_revert_seconds == -1:
                logger.info(f"[ErrorPro] 会话 {umo} 配置为不回退 Provider（-1）")

            # 切换后立即用新 Provider 重试用户原始提问
            if self.switch_retry_reply_enable:
                try:
                    success = await self._reply_with_switched_provider(event)
                    if success:
                        logger.info("[ErrorPro] 自动切换Provider后重试成功")
                        return True
                except Exception as e:
                    logger.error(f"[ErrorPro] 切换后自动重试回复失败: {e}")

            # 如果自动重试失败，给用户一个提示
            fallback_msg = f"已自动切换到备用服务，请重新尝试您的问题。"
            event.set_result(event.plain_result(fallback_msg))
            event.stop_event()
            return True

        except Exception as e:
            logger.error(f"[ErrorPro] 自动切换Provider失败: {e}")
            return False
