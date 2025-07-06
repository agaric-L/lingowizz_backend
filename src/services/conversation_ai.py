# conversation_ai.py

import os
import json
from typing import List, Dict, Any
# 导入智普AI的官方SDK
try:
    from zhipuai import ZhipuAI
except ImportError:
    print("ZhipuAI SDK not found. Please run 'pip install zhipuai'")
    ZhipuAI = None

class ConversationAIService:
    """
    AI对话服务。
    已重构为优先使用智普AI。如果未配置ZHIPU_API_KEY，则使用内置的备用逻辑。
    """
    
    def __init__(self):
        # 优先使用智普AI的API Key
        #self.zhipu_api_key = os.getenv('ZHIPU_API_KEY')
        self.zhipu_api_key ="482802cba1a144518285e3fb10068f4d.9drwfXmNcnr25cIe"
        self.conversation_history = {}  # 存储对话历史

        # 初始化智普AI客户端
        self.zhipu_client = None
        if self.zhipu_api_key and ZhipuAI:
            try:
                self.zhipu_client = ZhipuAI(api_key=self.zhipu_api_key)
            except Exception as e:
                print(f"Failed to initialize ZhipuAI client: {e}")
    
    def generate_conversation_themes(self, image_understanding: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据图片理解生成对话主题"""
        try:
            description = image_understanding.get('description', '')
            objects = image_understanding.get('objects', [])
            scene = image_understanding.get('scene', '')
            
            prompt = f"""
            Based on this image analysis:
            Description: {description}
            Objects: {', '.join(objects)}
            Scene: {scene}
            
            Generate 4 different conversation themes for language learning. Each theme must have:
            - A specific role for the AI (like Chef, Nutritionist, Shopping Assistant, Cultural Guide)
            - A clear scenario description
            - Educational value for vocabulary learning
            
            Your response MUST be a single, valid JSON array of objects. Each object must contain keys: "id", "title", "description", "role", "background", "scenario".
            """
            
            # 优先使用智普AI
            if self.zhipu_client:
                themes = self._generate_themes_zhipu(prompt)
            else:
                print("ZhipuAI client not available. Using fallback themes.")
                themes = self._generate_themes_fallback(image_understanding)
            
            return themes
            
        except Exception as e:
            print(f"Error generating conversation themes: {e}")
            return self._generate_themes_fallback(image_understanding)
    
    def _generate_themes_zhipu(self, prompt: str) -> List[Dict[str, Any]]:
        """使用智普AI SDK生成对话主题，并强制JSON输出"""
        if not self.zhipu_client:
            return []
        try:
            print("--- Calling Zhipu AI for theme generation ---")
            response = self.zhipu_client.chat.completions.create(
                model="glm-4-flash-250414",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},  # 强制JSON输出，非常稳定
                max_tokens=1024,
                temperature=0.7,
                stream=False
            )
            content = getattr(response, 'choices', [{}])[0].message.content if hasattr(response, 'choices') else None
            if not content:
                return []
            themes = json.loads(content)
            # 确保返回的是列表
            if isinstance(themes, dict) and isinstance(list(themes.values())[0], list):
                themes = list(themes.values())[0]

            return themes if isinstance(themes, list) else []
        except Exception as e:
            print(f"Error in Zhipu AI theme generation (SDK): {e}")
            return []

    def _generate_themes_fallback(self, image_understanding: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成备用对话主题"""
        scene = image_understanding.get('scene', 'general')
        if 'kitchen' in scene.lower():
            return [
                {'id': 1, 'title': 'Kitchen Cooking Assistant', 'description': '...', 'role': 'Chef', 'background': '...', 'scenario': '...'},
                {'id': 2, 'title': 'Healthy Eating Advisor', 'description': '...', 'role': 'Nutritionist', 'background': '...', 'scenario': '...'},
                {'id': 3, 'title': 'Grocery Shopping Helper', 'description': '...', 'role': 'Shopping Assistant', 'background': '...', 'scenario': '...'},
                {'id': 4, 'title': 'Food Culture Explorer', 'description': '...', 'role': 'Cultural Guide', 'background': '...', 'scenario': '...'}
            ]
        return [{'id': 1, 'title': 'General Learning Assistant', 'description': '...', 'role': 'Teacher', 'background': '...', 'scenario': '...'}]

    def generate_ai_response(self, user_message: str, session_context: Dict[str, Any]) -> str:
        """生成AI回复（非流式）"""
        try:
            session_id = session_context.get('session_id')
            role = session_context.get('role', 'Assistant')
            theme = session_context.get('theme', 'General Chat')
            background = session_context.get('background', '')
            
            history = self.conversation_history.get(session_id, [])
            
            if self.zhipu_client:
                response_text = self._generate_response_zhipu(user_message, role, theme, background, history)
            else:
                print("ZhipuAI client not available. Using fallback response.")
                response_text = self._generate_response_fallback(user_message, role, theme)
            
            # 更新对话历史
            if session_id:
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []
                self.conversation_history[session_id].append({'user': user_message, 'assistant': response_text})
                if len(self.conversation_history[session_id]) > 10:
                    self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
            
            return response_text
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return self._generate_response_fallback(user_message, 'Assistant', 'General Chat')
    
    def _generate_response_zhipu(self, user_message: str, role: str, theme: str, 
                                background: str, history: List[Dict]) -> str:
        """使用智普AI SDK生成回复"""
        if not self.zhipu_client:
            return self._generate_response_fallback(user_message, role, theme)
        try:
            print(f"--- Calling Zhipu AI for conversation (Role: {role}) ---")
            system_prompt = f"""
            You are a {role} in a {theme} scenario. {background}
            Your goal is to help the user learn English vocabulary through natural conversation.
            - Stay in character as a {role}.
            - Use vocabulary appropriate for the scenario.
            - Provide helpful explanations when needed.
            - Keep responses conversational and engaging.
            - If the user asks about vocabulary, provide clear definitions and examples.
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-5:]:  # 只保留最近5轮对话
                messages.append({"role": "user", "content": h['user']})
                messages.append({"role": "assistant", "content": h['assistant']})
            messages.append({"role": "user", "content": user_message})
            
            response = self.zhipu_client.chat.completions.create(
                model="glm-4-flash-250414",
                messages=messages,
                max_tokens=300,
                temperature=0.7,
                stream=False
            )
            content = getattr(response, 'choices', [{}])[0].message.content if hasattr(response, 'choices') else None
            return content if content else self._generate_response_fallback(user_message, role, theme)
                
        except Exception as e:
            print(f"Error in Zhipu AI response generation: {e}")
            return self._generate_response_fallback(user_message, role, theme)

    def _generate_response_fallback(self, user_message: str, role: str, theme: str) -> str:
        """生成备用回复"""
        return f"As a {role}, regarding '{user_message}', let's discuss this further in our '{theme}' scenario."
    
    def clear_conversation_history(self, session_id: str):
        """清除对话历史"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            print(f"History for session {session_id} cleared.")

# 创建全局实例
conversation_ai_service = ConversationAIService()
# ==============================================================================
#  STANDALONE TEST SCRIPT
#  To run, execute: python conversation_ai.py
#  Make sure to set the ZHIPU_API_KEY environment variable first.
# ==============================================================================

if __name__ == "__main__":
    
    print("=" * 50)
    print("  STARTING STANDALONE TEST FOR ConversationAIService")
    print("=" * 50)

    # 1. 检查API Key是否设置
    api_key = "482802cba1a144518285e3fb10068f4d.9drwfXmNcnr25cIe" # 直接硬编码
    if not api_key:
        print("\nERROR: ZHIPU_API_KEY environment variable is not set.")
        print("Please set it before running the test.")
        print("Example: export ZHIPU_API_KEY='your_key_here'\n")
    else:
        # 2. 初始化服务
        service = ConversationAIService()

        # 3. 测试主题生成功能 (generate_conversation_themes)
        print("\n--- TEST 1: Generating Conversation Themes ---")
        mock_image_data = {
            'description': 'A modern, clean kitchen with various fruits on a wooden countertop.',
            'objects': ['apple', 'banana', 'orange', 'knife', 'cutting board'],
            'scene': 'kitchen'
        }
        print(f"Input image understanding: {mock_image_data}")
        
        themes = service.generate_conversation_themes(mock_image_data)
        
        if themes:
            print("\nSuccessfully generated themes:")
            # 使用json.dumps美化输出
            print(json.dumps(themes, indent=2))
        else:
            print("\nFailed to generate themes. Fallback might have been used.")
        
        print("\n--- END OF TEST 1 ---")

        # 4. 测试对话生成功能 (generate_ai_response)
        if themes:
            print("\n--- TEST 2: Simulating a Conversation ---")
            
            # 选择第一个生成的主题来开始对话
            selected_theme = themes[0]
            session_context = {
                'session_id': 'test-session-123',
                'role': selected_theme.get('role', 'Assistant'),
                'theme': selected_theme.get('title', 'General Chat'),
                'background': selected_theme.get('background', '')
            }
            
            print(f"\nStarting conversation with context: {session_context}\n")
            
            # 对话第一轮
            user_message_1 = "Hello! I see some fruits. What can you tell me about them?"
            print(f"User > {user_message_1}")
            ai_response_1 = service.generate_ai_response(user_message_1, session_context)
            print(f"AI ({session_context['role']}) > {ai_response_1}\n")
            
            # 对话第二轮 (测试历史记录)
            user_message_2 = "That's helpful. How do you use a 'cutting board'?"
            print(f"User > {user_message_2}")
            ai_response_2 = service.generate_ai_response(user_message_2, session_context)
            print(f"AI ({session_context['role']}) > {ai_response_2}\n")

            # 检查历史记录是否被保存
            print("Current conversation history:")
            print(json.dumps(service.conversation_history.get(session_context['session_id']), indent=2))

            # 清除历史记录
            service.clear_conversation_history(session_context['session_id'])
            print(f"History state after clearing: {service.conversation_history}")

            print("\n--- END OF TEST 2 ---")
        else:
            print("\nSkipping conversation test because no themes were generated.")

    print("\n=" * 50)
    print("  TEST SCRIPT FINISHED")
    print("=" * 50)