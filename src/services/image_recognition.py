# image_recognition.py

import os
import json
import base64
from typing import List, Dict, Any

# 图像处理和模型库
import cv2
from PIL import Image  # Pillow库对于显示图片至关重要
import torch

# ultralytics 用于 YOLOv8 模型
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics library not found. Please run 'pip install ultralytics opencv-python'")
    YOLO = None

# zhipuai 用于多模态理解和单词定义
try:
    from zhipuai import ZhipuAI
except ImportError:
    print("ZhipuAI SDK not found. Please run 'pip install zhipuai'")
    ZhipuAI = None


class ImageRecognitionService:
    """
    图片识别和AI服务。
    - 使用 智普AI GLM-4V 进行图片理解 (生成结构化描述)。
    - 使用 YOLOv8 进行物体检测和分割。
    - 使用 智普AI GLM-4 进行单词定义。
    """
    
    def __init__(self):
        # 配置API密钥
        #self.zhipu_api_key = os.getenv('ZHIPU_API_KEY', "482802cba1a144518285e3fb10068f4d.9drwfXmNcnr25cIe")
        self.zhipu_api_key="482802cba1a144518285e3fb10068f4d.9drwfXmNcnr25cIe"
        # --- 智普AI客户端初始化 ---
        self.zhipu_client = None
        if self.zhipu_api_key and ZhipuAI:
            self.zhipu_client = ZhipuAI(api_key=self.zhipu_api_key)
            print("ZhipuAI client initialized.")

        # --- YOLOv8模型初始化 ---
        self.yolo_model = None
        if YOLO:
            try:
                print("Loading YOLOv8 model...")
                self.yolo_model = YOLO("yolov8n.pt")
                print("YOLOv8 model loaded successfully.")
            except Exception as e:
                print(f"Failed to load YOLOv8 model: {e}")

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
            
            try:
                content = response.choices[0].message.content  # type: ignore
            except (AttributeError, IndexError):
                print("GLM-4V: Streaming response received, using fallback")
                return self._get_fallback_understanding()
            if not content:
                print("GLM-4V: No content received")
                return self._get_fallback_understanding()
            try:
                cleaned_content = content.strip().lstrip("```json").rstrip("```").strip()
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                print(f"GLM-4V: Failed to decode JSON. Content: {content}")
                return {'description': content, 'objects': [], 'scene': 'unknown', 'mood': 'neutral'}

        except Exception as e:
            print(f"Error in GLM-4V image understanding: {e}")
            return self._get_fallback_understanding()

    def segment_objects_yolo(self, image_path: str, upload_folder: str) -> List[Dict[str, Any]]:
        if not self.yolo_model:
            print("YOLOv8 model is not available. Cannot perform object segmentation.")
            return []
        try:
            results = self.yolo_model(image_path, conf=0.4)
            detected_objects = []
            if results:
                result = results[0] 
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                class_names = result.names
                for i, bbox in enumerate(boxes):
                    detected_objects.append({
                        'id': i + 1, 'name': class_names[int(class_ids[i])],
                        'confidence': float(confidences[i]), 'bbox': [int(coord) for coord in bbox]
                    })
            return self._create_segmented_images(image_path, detected_objects, upload_folder)
        except Exception as e:
            print(f"Error in YOLOv8 object segmentation: {e}")
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
                relative_path = os.path.join('/static/uploads', segmented_filename).replace('\\', '/')
                segmented_objects.append({
                    **obj, 'segmented_image_path_local': segmented_path, 'segmented_image': relative_path 
                })
            return segmented_objects
        except Exception as e:
            print(f"Error creating segmented images: {e}")
            return []
    
    def generate_word_definition_zhipu(self, word: str) -> Dict[str, Any]:
        if not self.zhipu_client:
            print("ZhipuAI client is not available. Using fallback for word definition.")
            return self._get_fallback_word_info(word)
        try:
            response = self.zhipu_client.chat.completions.create(
                model="glm-4-flash-250414",
                messages=[{
                    "role": "user",
                    "content": f"Please provide a definition and example sentence for the word '{word}'. Your response MUST be a single, valid JSON object with keys: \"word\", \"definition\", \"example_sentence\", \"pronunciation\", and \"part_of_speech\"."
                }],
                response_format={"type": "json_object"}, 
                max_tokens=250,
                stream=False
            )
            try:
                content = response.choices[0].message.content  # type: ignore
            except (AttributeError, IndexError):
                print("ZhipuAI: Streaming response received, using fallback")
                return self._get_fallback_word_info(word)
            if not content:
                print("ZhipuAI: No content received for word definition")
                return self._get_fallback_word_info(word)
            return json.loads(content)
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
#  To run, execute: python image_recognition.py
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

        # --- TEST 1: 图片理解 (GLM-4V) ---
        print("\n--- TEST 1: Generating Image Understanding with GLM-4V ---")
        understanding_result = service.understand_image_glm4v(test_image_path)
        print("GLM-4V Understanding Result:")
        print(json.dumps(understanding_result, indent=2))
        print("--- END OF TEST 1 ---\n")

        # --- TEST 2: 物体检测与分割 (YOLOv8) ---
        print("\n--- TEST 2: Segmenting Objects with YOLOv8 ---")
        yolo_objects = service.segment_objects_yolo(test_image_path, upload_folder=upload_dir)
        
        if yolo_objects:
            print(f"Successfully detected and segmented {len(yolo_objects)} objects with YOLOv8.")
            print("YOLOv8 Detected Objects:")
            print(json.dumps(yolo_objects, indent=2, ensure_ascii=False))
            print(f"\nSegmented images are saved in '{upload_dir}'")

            # =====================================================
            #  新功能：可视化展示分割后的图片
            # =====================================================
            print("\n--- VISUAL CONFIRMATION: Displaying Segmented Images ---")
            print("Each detected object will be opened in your default image viewer.")
            print("Close the image viewer window to continue to the next one.")

            for i, obj in enumerate(yolo_objects):
                try:
                    # 获取已保存的分割图片的本地路径
                    img_path = obj['segmented_image_path_local']
                    
                    if os.path.exists(img_path):
                        # 使用Pillow库打开并显示图片
                        img = Image.open(img_path)
                        # .show() 会调用系统的默认图片查看器
                        img.show(title=f"Segmented: {obj['name']}")
                        
                        print(f"\nDisplaying object {i+1}/{len(yolo_objects)}: '{obj['name']}' (Confidence: {obj['confidence']:.2f})")
                        
                        # 暂停，等待用户按回车继续，防止窗口一次性全部弹出
                        if i < len(yolo_objects) - 1:
                            input("-> Press Enter to view the next segmented image...")
                        else:
                            print("\nAll segmented images have been shown.")
                    else:
                        print(f"Warning: Image file not found at {img_path}")

                except Exception as e:
                    print(f"Could not display image {obj.get('segmented_image_path_local', 'N/A')}. Error: {e}")
            # =====================================================

        else:
            print("No objects were detected by YOLOv8.")
        
        print("--- END OF TEST 2 ---\n")
        
        # --- 最终整合结果 ---
        if yolo_objects:
            glm_objects = set(understanding_result.get('objects', []))
            yolo_names = set(obj['name'] for obj in yolo_objects)
            combined_objects = list(glm_objects.union(yolo_names))
            understanding_result['objects'] = combined_objects
        
        print("\n--- FINAL COMBINED RESULT (for ConversationService) ---")
        print("This combined dictionary can be passed to the ConversationAIService.")
        print(json.dumps(understanding_result, indent=2, ensure_ascii=False))
        
        # --- TEST 3: 单词定义 (ZhipuAI) ---
        if yolo_objects:
            first_object_name = yolo_objects[0]['name']
            print(f"\n--- TEST 3: Generating Word Definition for '{first_object_name}' with ZhipuAI ---")
            
            word_info = service.generate_word_definition_zhipu(first_object_name)
            print("Definition from ZhipuAI:")
            print(json.dumps(word_info, indent=2, ensure_ascii=False))
            
            print("--- END OF TEST 3 ---\n")

    print("\n" + "="*60)
    print("  TEST SCRIPT FINISHED")
    print("="*60 + "\n")