import os
import json
import base64
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass

# --- New imports for Vivo BlueLM API call ---
import uuid
import time
import hashlib
import hmac
from urllib.parse import quote
import requests

# 图像处理和模型库
import cv2
from PIL import Image

import random

# ultralytics 用于 YOLOv8 模型
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics library not found. Please run 'pip install ultralytics opencv-python'")
    YOLO = None

# zhipuai 用于多模态理解和单词定义 (作为备用)
try:
    from zhipuai import ZhipuAI
except ImportError:
    print("ZhipuAI SDK not found. Please run 'pip install zhipuai'")
    ZhipuAI = None

@dataclass
class ZhipuMessage:
    content: str

@dataclass
class ZhipuChoice:
    message: ZhipuMessage

@dataclass
class ZhipuResponse:
    choices: List[ZhipuChoice]

# ==============================================================================
#  VIVO BLUE-LM API AUTHENTICATION HELPERS
# ==============================================================================
def _gen_nonce(length: int = 16) -> str:
    """生成指定长度的随机字符串。"""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(chars) for _ in range(length))

def _gen_sign_headers(app_id: str, app_key: str, method: str, uri: str, params: Dict[str, Any]) -> Dict[str, str]:
    """生成调用Vivo AI API所需的认证头，使用官方指定的HMAC-SHA256签名算法。"""
    timestamp = str(int(time.time()))
    nonce = _gen_nonce()

    if not params:
        canonical_query_string = ""
    else:
        sorted_params = sorted(params.items())
        encoded_parts = [f"{quote(k)}={quote(str(v))}" for k, v in sorted_params]
        canonical_query_string = "&".join(encoded_parts)

    signed_headers_string = (f"x-ai-gateway-app-id:{app_id}\n"
                           f"x-ai-gateway-timestamp:{timestamp}\n"
                           f"x-ai-gateway-nonce:{nonce}")

    signing_string = (f"{method}\n{uri}\n{canonical_query_string}\n{app_id}\n"
                      f"{timestamp}\n{signed_headers_string}")

    hashed = hmac.new(app_key.encode('utf-8'), signing_string.encode('utf-8'), hashlib.sha256).digest()
    signature = base64.b64encode(hashed).decode('utf-8')
    
    return {
        'Content-Type': 'application/json',
        'X-AI-GATEWAY-APP-ID': "2025765980",
        'X-AI-GATEWAY-TIMESTAMP': timestamp,
        'X-AI-GATEWAY-NONCE': nonce,
        'X-AI-GATEWAY-SIGNED-HEADERS': 'x-ai-gateway-app-id;x-ai-gateway-timestamp;x-ai-gateway-nonce',
        'X-AI-GATEWAY-SIGNATURE': signature
    }

class ImageRecognitionService:
    """
    图片识别和AI服务。
    - 优先使用蓝心大模型进行单词定义。
    - 使用 智普AI GLM-4V 进行图片理解 (生成结构化描述)。
    - 使用 YOLOv8 进行物体检测和分割。
    """
    
    def __init__(self):
        # --- 智普AI客户端初始化 (用于图片理解和备用) ---
        self.zhipu_api_key = os.getenv('ZHIPU_API_KEY', "482802cba1a144518285e3fb10068f4d.9drwfXmNcnr25cIe")
        self.zhipu_client = None
        if self.zhipu_api_key and ZhipuAI:
            self.zhipu_client = ZhipuAI(api_key=self.zhipu_api_key)
            print("ZhipuAI client initialized.")
            
        # --- 蓝心大模型 (Vivo BlueLM) 配置 (用于单词定义) ---
        self.vivo_app_id = os.getenv('VIVO_APP_ID')
        self.vivo_app_key = os.getenv('VIVO_APP_KEY')
        if not self.vivo_app_id or not self.vivo_app_key:
            self.vivo_app_id = "2025765980"  # 替换为你的 APP_ID
            self.vivo_app_key = "sQUHXkeeXJGaxQQJ" # 替换为你的 APP_KEY
        self.vivo_base_url = "https://api-ai.vivo.com.cn"
        self.vivo_uri = "/vivogpt/completions"
        self.vivo_api_url = f"{self.vivo_base_url}{self.vivo_uri}"
        self.vivo_model_name = "vivo-BlueLM-TB-Pro"
        if self.vivo_app_id != "YOUR_VIVO_APP_ID":
            print("Vivo BlueLM client configured.")

        # --- (新) Hugging Face API 配置 ---
        # 从环境变量中读取你的API URL和Token，这是最佳实践
        self.hf_api_url = os.getenv('HF_API_URL')
        self.hf_api_token = os.getenv('HF_API_TOKEN')
        
        if self.hf_api_url and self.hf_api_token:
            print("Hugging Face Inference API configured for object detection.")
        else:
            print("WARNING: Hugging Face API URL or Token not found in environment variables.")
            print("Please set HF_API_URL and HF_API_TOKEN to use object detection.")

    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def understand_image_glm4v(self, image_path: str) -> Dict[str, Any]:
        """使用智普AI GLM-4V进行图片理解，并要求返回JSON"""
        if not self.zhipu_client:
            print("ZhipuAI client is not available. Using fallback understanding.")
            return self._get_fallback_understanding()

        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            prompt = """
            Analyze the image provided and return a structured JSON object.
            The JSON object must contain the following keys:
            1. "description": A concise, one-sentence summary of the image.
            2. "objects": A list of strings, naming the key objects visible in the image.
            3. "scene": A short phrase describing the overall scene or environment (e.g., "city street at night", "kitchen countertop", "beach on a sunny day").
            4. "mood": A single word describing the mood or atmosphere of the image (e.g., "peaceful", "energetic", "somber").
            
            Your response MUST be only the valid JSON object, without any surrounding text or markdown formatting.
            """
            
            response = self.zhipu_client.chat.completions.create(
                model="glm-4v-flash", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_image}}
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.7,
                stream=False
            )
            
            # Extract response content from ZhipuAI response
            try:
                # Convert response to dict if needed
                response_dict = response if isinstance(response, dict) else response.__dict__
                
                # Extract content safely
                content = ''
                try:
                    content = response_dict.get('choices', [{}])[0].get('message', {}).get('content', '')
                except (AttributeError, IndexError, TypeError):
                    content = str(response)
                
                if not content:
                    return self._get_fallback_understanding()
                
                # Clean and parse JSON
                cleaned_content = content.strip().lstrip("```json").rstrip("```").strip()
                return json.loads(cleaned_content)
            except Exception as e:
                print(f"Error processing response: {e}")
                return self._get_fallback_understanding()

        except Exception as e:
            print(f"Error in GLM-4V image understanding: {e}")
            return self._get_fallback_understanding()

    def segment_objects_via_api(self, image_path: str, upload_folder: str) -> List[Dict[str, Any]]:
        """
        通过调用Hugging Face API进行物体检测，然后裁剪出分割后的图片。
        """
        if not self.hf_api_url or not self.hf_api_token:
            print("Hugging Face API is not configured. Cannot perform object segmentation.")
            return []
            
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        
        try:
            # 1. 读取图片数据并发送API请求
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            print("Sending image to Hugging Face API for segmentation...")
            api_response = requests.post(self.hf_api_url, headers=headers, data=image_data, timeout=30)
            api_response.raise_for_status()  # 如果请求失败 (如 4xx 或 5xx), 会抛出异常
            
            results_json = api_response.json()
            print("Received response from Hugging Face API.")

            # 2. 将API返回的JSON解析成我们需要的格式
            detected_objects = []
            if not isinstance(results_json, list):
                print(f"Error: API response is not a list, but {type(results_json)}. Response: {results_json}")
                return []

            for i, item in enumerate(results_json):
                # Hugging Face API的返回格式通常是 {'score': 0.9, 'label': 'cat', 'box': {'xmin': 10, ...}}
                box = item.get('box')
                if not box: continue

                detected_objects.append({
                    'id': i + 1,
                    'name': item.get('label', 'unknown'),
                    'confidence': float(item.get('score', 0.0)),
                    'bbox': [
                        int(box.get('xmin', 0)),
                        int(box.get('ymin', 0)),
                        int(box.get('xmax', 0)),
                        int(box.get('ymax', 0))
                    ]
                })

            return self._create_segmented_images(image_path, detected_objects, upload_folder)

        except requests.exceptions.RequestException as e:
            print(f"Error calling Hugging Face API: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during API-based segmentation: {e}")
            return []
    
    def _create_segmented_images(self, image_path: str, objects: List[Dict], upload_folder: str) -> List[Dict[str, Any]]:
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder, exist_ok=True)
        try:
            image = cv2.imread(image_path)
            segmented_objects = []
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                cropped_image = image[y1:y2, x1:x2]
                if cropped_image.size == 0: continue
                obj_name_safe = obj['name'].replace(' ', '_')
                segmented_filename = f"segmented_{obj_name_safe}_{obj['id']}.png"
                segmented_path = os.path.join(upload_folder, segmented_filename)
                cv2.imwrite(segmented_path, cropped_image)
                updated_obj = obj.copy()
                updated_obj.update({
                    'segmented_image_path_local': segmented_path,
                    'segmented_image': os.path.join('/static', os.path.basename(upload_folder), segmented_filename).replace('\\', '/')
                })
                segmented_objects.append(updated_obj)
            return segmented_objects
        except Exception as e:
            print(f"Error creating segmented images: {e}")
            return []

    def _identify_single_object_glm4v(self, image_path: str) -> str:
        """使用GLM-4V识别单个裁剪后的图片中的物体名称"""
        if not self.zhipu_client:
            return "unknown"
        try:
            base64_image = self.encode_image_to_base64(image_path)
            prompt = "What is the single, primary object in this image? Respond with ONLY a single word or short phrase, without any extra text."
            
            response = self.zhipu_client.chat.completions.create(
                model="glm-4v-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_image}}
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.1,
                stream=False
            )
            # Handle response based on type
            try:
                response_dict = response if isinstance(response, dict) else response.__dict__
                content = response_dict.get('choices', [{}])[0].get('message', {}).get('content', '')
                return content.strip().lower() if content else "unknown"
            except Exception:
                return "unknown"
        except Exception as e:
            print(f"Error identifying single object with GLM-4V: {e}")
            return "unknown"

    def segment_and_identify_combined(self, image_path: str, upload_folder: str) -> List[Dict[str, Any]]:
        """
        第一步: 使用Hugging Face API进行快速分割。
        第二步: 对每个小图，使用GLM-4V进行精确识别。
        """
        detected_objects = self.segment_objects_via_api(image_path, upload_folder)
        
        if not detected_objects: 
            print("No objects detected via API. Stopping.")
            return []
        
        # 后续的识别流程保持不变
        final_identified_objects = []
        for obj in detected_objects:
            local_path = obj.get('segmented_image_path_local')
            if not local_path or not os.path.exists(local_path): continue
            
            glm4v_name = self._identify_single_object_glm4v(local_path)
            final_obj = obj.copy()
            final_obj['name'] = glm4v_name
            final_identified_objects.append(final_obj)
        return final_identified_objects
        
    def generate_word_definition(self, word: str) -> Dict[str, Any]:
        """
        生成单词定义。优先使用蓝心大模型，如果未配置则使用智普AI作为备用。
        """
        if self.vivo_app_id and self.vivo_app_id != "YOUR_VIVO_APP_ID":
            print(f"--- Generating definition for '{word}' using Vivo BlueLM ---")
            return self._generate_word_definition_vivo(word)
        elif self.zhipu_client:
            print(f"--- Generating definition for '{word}' using ZhipuAI (fallback) ---")
            return self._generate_word_definition_zhipu(word)
        else:
            print(f"--- No AI client configured. Using fallback for '{word}' definition. ---")
            return self._get_fallback_word_info(word)

    def _generate_word_definition_vivo(self, word: str) -> Dict[str, Any]:
        """使用蓝心大模型API生成单词定义。"""
        prompt = (f"Please provide a definition and example sentence for the English word '{word}'. "
                  f"Your response MUST be a single, valid JSON object with keys: "
                  f"\"word\", \"definition\", \"example_sentence\", \"pronunciation\", and \"part_of_speech\". "
                  f"Do not include any text outside of the JSON object.")
        
        if not self.vivo_app_id or not self.vivo_app_key:
            raise ValueError("Vivo API credentials not properly configured")
            
        request_id = str(uuid.uuid4())
        params = {'requestId': request_id}
        headers = _gen_sign_headers(str(self.vivo_app_id), str(self.vivo_app_key), "POST", self.vivo_uri, params)
        
        payload = {
            'prompt': prompt,
            'model': self.vivo_model_name,
            'sessionId': str(uuid.uuid4()),
            'extra': {'temperature': 0.5, 'max_new_tokens': 250}
        }
        
        content = ""
        try:
            response = requests.post(self.vivo_api_url, json=payload, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            res_json = response.json()

            if res_json.get('code') != 0:
                print(f"Vivo API Error - Code: {res_json.get('code')}, Msg: {res_json.get('msg')}")
                return self._get_fallback_word_info(word)
            
            content = res_json.get('data', {}).get('content', '')
            if not content:
                print("Vivo API returned empty content.")
                return self._get_fallback_word_info(word)
            
            # 清理模型可能返回的Markdown代码块
            cleaned_content = content.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(cleaned_content)
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Vivo API: {e}")
            return self._get_fallback_word_info(word)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from Vivo API. Content: {content}")
            return self._get_fallback_word_info(word)
        except Exception as e:
            print(f"An unexpected error occurred during Vivo word definition: {e}")
            return self._get_fallback_word_info(word)
    
    def _generate_word_definition_zhipu(self, word: str) -> Dict[str, Any]:
        """使用智普AI作为备用方案生成单词定义。"""
        if not self.zhipu_client:
            return self._get_fallback_word_info(word)
            
        try:
            response = self.zhipu_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{
                    "role": "user",
                    "content": f"Please provide a definition and example sentence for the word '{word}'. Your response MUST be a single, valid JSON object with keys: \"word\", \"definition\", \"example_sentence\", \"pronunciation\", and \"part_of_speech\"."
                }],
                response_format={"type": "json_object"}, 
                max_tokens=250,
                stream=False
            )
            # Handle response based on type
            try:
                response_dict = response if isinstance(response, dict) else response.__dict__
                content = response_dict.get('choices', [{}])[0].get('message', {}).get('content', '')
                return json.loads(content) if content else self._get_fallback_word_info(word)
            except Exception:
                return self._get_fallback_word_info(word)
        except Exception as e:
            print(f"Error generating word definition with ZhipuAI: {e}")
            return self._get_fallback_word_info(word)
    
    def _get_fallback_understanding(self) -> Dict[str, Any]:
        return { 'description': 'Could not describe the image. It contains various objects for learning.', 'objects': [], 'scene': 'general', 'mood': 'educational' }
    
    def _get_fallback_word_info(self, word: str) -> Dict[str, Any]:
        return { 'word': word, 'definition': 'A common object found in everyday life.', 'example_sentence': f'I see a {word}.', 'pronunciation': f'/{word}/', 'part_of_speech': 'noun' }

# 创建全局实例
image_recognition_service = ImageRecognitionService()

# ==============================================================================
#  STANDALONE TEST SCRIPT
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  STARTING STANDALONE TEST FOR ImageRecognitionService")
    print("="*60 + "\n")

    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    upload_dir = os.path.join(test_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    test_image_path = os.path.join(test_dir, "test_image2.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at '{test_image_path}'")
        print("Please place a JPEG image named 'test_image2.jpg' in the 'test_images' directory to run the test.")
    else:
        service = image_recognition_service

        # --- TEST 1: Image Understanding (GLM-4V) ---
        print("\n--- TEST 1: Image Understanding (GLM-4V) ---")
        understanding = service.understand_image_glm4v(test_image_path)
        print(json.dumps(understanding, indent=2, ensure_ascii=False))
        print("--- END OF TEST 1 ---")
        
        # --- TEST 2: Combined Segmentation & Identification ---
        print("\n--- TEST 2: Combined Segmentation (YOLOv8) and Identification (GLM-4V) ---")
        combined_results = service.segment_and_identify_combined(test_image_path, upload_folder=upload_dir)
        if combined_results:
            print(json.dumps(combined_results, indent=2, ensure_ascii=False))
        else:
            print("The combined process did not yield any results.")
        print("--- END OF TEST 2 ---")
        
        # --- TEST 3: Word Definition ---
        print("\n--- TEST 3: Word Definition (Vivo BlueLM or ZhipuAI Fallback) ---")
        if (service.vivo_app_id == "YOUR_VIVO_APP_ID" and not service.zhipu_client):
            print("SKIPPING: Neither Vivo nor ZhipuAI API is configured.")
        else:
            test_word = "banana" # A word likely to be in the test image
            word_info = service.generate_word_definition(test_word)
            print(json.dumps(word_info, indent=2, ensure_ascii=False))
        print("--- END OF TEST 3 ---")

    print("\n" + "="*60)
    print("  TEST SCRIPT FINISHED")
    print("="*60 + "\n")
