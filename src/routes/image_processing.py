from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
from ..services.image_recognition2 import  image_recognition_service
from ..services.conversation_ai import conversation_ai_service

#这个文件是项目的核心功能，是对图像处理相关功能的集成
'''
这个文件是 LingoWizz 项目中非常核心和有趣的一部分，后端业务逻辑封装
但是这里面的路游调用的功能是在哪里写的image_recognition_service、conversation_ai_service
它连接了前端用户操作、后端文件处理、以及强大的 AI 图像识别和理解服务。
代码逻辑清晰，将不同的AI功能拆分到独立的API端点中，这是一种很好的设计。
'''

image_bp = Blueprint('image', __name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#上传图片
@image_bp.route('/upload-image', methods=['POST'])
def upload_image():
    """上传图片"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # 确保上传目录存在 - 修正为与Flask静态文件目录一致
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), UPLOAD_FOLDER)
            os.makedirs(upload_dir, exist_ok=True)
            
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filepath': filepath,
                'filename': filename
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/segment-objects', methods=['POST'])
def segment_objects():
    """物品识别和分割 - YOLO分割 + GLM-4V精确识别"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')

        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Invalid image path'}), 400

        # --- 添加缺失的逻辑 ---
        # 定义上传目录的绝对路径 - 修正为与Flask静态文件目录一致
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), UPLOAD_FOLDER)
        os.makedirs(upload_dir, exist_ok=True)
        # --- 结束添加 ---

        # 使用YOLO分割 + GLM-4V精确识别的组合方法
        objects = image_recognition_service.segment_and_identify_combined(image_path, upload_folder=upload_dir)

        return jsonify({
            'success': True,
            'objects': objects
        })

    except Exception as e:
        # 打印详细错误到后端控制台，方便调试
        print(f"Error in /segment-objects: {e}") 
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/generate-word-info', methods=['POST'])
def generate_word_info():
    """生成单词信息"""
    try:
        data = request.get_json()
        word = data.get('word')
        
        if not word:
            return jsonify({'success': False, 'error': 'Word is required'}), 400
        
        # 使用OpenAI生成单词信息
        word_info = image_recognition_service.generate_word_definition_zhipu(word)
        
        return jsonify({
            'success': True,
            'word_info': word_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/understand-image', methods=['POST'])
def understand_image():
    """图片理解"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Invalid image path'}), 400
        
        # 使用OpenAI GPT-4V进行图片理解
        understanding = image_recognition_service.understand_image_glm4v(image_path)
        
        return jsonify({
            'success': True,
            'understanding': understanding
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/generate-conversation-themes', methods=['POST'])
def generate_conversation_themes():
    """生成对话主题"""
    try:
        data = request.get_json()
        understanding = data.get('understanding')
        
        if not understanding:
            return jsonify({'success': False, 'error': 'Image understanding is required'}), 400
        
        # 使用AI服务生成对话主题
        themes = conversation_ai_service.generate_conversation_themes(understanding)
        
        return jsonify({
            'success': True,
            'themes': themes
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

