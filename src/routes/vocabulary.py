from flask import Blueprint, request, jsonify
from src.models.vocabulary import VocabularyItem
from src.models import db             # <-- 从中央位置导入 db
import json

'''
单词本的相关操作
GET /vocabulary (get_vocabulary):
目的: 获取单词本中的所有单词，并支持分页。
流程: It uses request.args to get page and per_page parameters. The paginate() function from SQLAlchemy is used, which is the most efficient way to handle pagination. It returns a well-structured JSON object containing the list of words for the current page, total number of words, total pages, and the current page number. This is perfect for a front-end UI.
POST /vocabulary (add_vocabulary):
目的: 向单词本添加一个新单词。
流程: It validates that required fields (word, definition) are present. It then checks if the word already exists to prevent duplicates, which is great for data integrity. If the word is new, it creates a VocabularyItem, saves it to the database, and returns the new item with a 201 Created status code.
GET /vocabulary/<int:item_id> (get_vocabulary_item):
目的: 获取单个单词的详细信息。
流程: Uses query.get_or_404(item_id) to find the item, which is a clean and robust way to handle both success and "not found" cases.
PUT /vocabulary/<int:item_id> (update_vocabulary_item):
目的: 更新一个已存在的单词条目。
流程: Finds the item using get_or_404, then checks the incoming JSON for fields to update. This allows for partial updates (like a PATCH request).
DELETE /vocabulary/<int:item_id> (delete_vocabulary_item):
目的: 从单词本中删除一个单词。
流程: Finds the item, deletes it from the database session, and commits the change.
GET /vocabulary/search (search_vocabulary):
目的: 根据关键词搜索单词本。
流程: It takes a query parameter q and uses filter(VocabularyItem.word.contains(query) | VocabularyItem.definition.contains(query)) to search for the query string in both the word and its definition. The | operator correctly translates to a SQL OR.
GET /vocabulary/export (export_vocabulary):
目的: 将整个单词本导出为 JSON 格式。
流程: It fetches all vocabulary items and wraps them in a JSON object with some metadata like total count and an export timestamp.
'''

vocabulary_bp = Blueprint('vocabulary', __name__)

@vocabulary_bp.route('/vocabulary', methods=['GET'])
def get_vocabulary():
    """获取所有单词"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        vocabulary_items = VocabularyItem.query.order_by(VocabularyItem.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'vocabulary': [item.to_dict() for item in vocabulary_items.items],
            'total': vocabulary_items.total,
            'pages': vocabulary_items.pages,
            'current_page': page
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@vocabulary_bp.route('/vocabulary', methods=['POST'])
def add_vocabulary():
    """添加新单词到单词本"""
    try:
        data = request.get_json()
        
        required_fields = ['word', 'definition']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必需字段: {field}'}), 400
        
        # 检查单词是否已存在
        existing_word = VocabularyItem.query.filter_by(word=data['word']).first()
        if existing_word:
            return jsonify({'error': '该单词已存在于单词本中'}), 400
        
        vocabulary_item = VocabularyItem(
            word=data['word'],
            definition=data['definition'],
            example_sentence=data.get('example_sentence'),
            image_path=data.get('image_path'),
            segmented_image_path=data.get('segmented_image_path')
        )
        
        db.session.add(vocabulary_item)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'vocabulary_item': vocabulary_item.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@vocabulary_bp.route('/vocabulary/<int:item_id>', methods=['GET'])
def get_vocabulary_item(item_id):
    """获取单个单词详情"""
    try:
        vocabulary_item = VocabularyItem.query.get_or_404(item_id)
        return jsonify({
            'success': True,
            'vocabulary_item': vocabulary_item.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@vocabulary_bp.route('/vocabulary/<int:item_id>', methods=['PUT'])
def update_vocabulary_item(item_id):
    """更新单词信息"""
    try:
        vocabulary_item = VocabularyItem.query.get_or_404(item_id)
        data = request.get_json()
        
        # 更新字段
        if 'word' in data:
            vocabulary_item.word = data['word']
        if 'definition' in data:
            vocabulary_item.definition = data['definition']
        if 'example_sentence' in data:
            vocabulary_item.example_sentence = data['example_sentence']
        if 'image_path' in data:
            vocabulary_item.image_path = data['image_path']
        if 'segmented_image_path' in data:
            vocabulary_item.segmented_image_path = data['segmented_image_path']
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'vocabulary_item': vocabulary_item.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@vocabulary_bp.route('/vocabulary/<int:item_id>', methods=['DELETE'])
def delete_vocabulary_item(item_id):
    """删除单词"""
    try:
        vocabulary_item = VocabularyItem.query.get_or_404(item_id)
        db.session.delete(vocabulary_item)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '单词已删除'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@vocabulary_bp.route('/vocabulary/search', methods=['GET'])
def search_vocabulary():
    """搜索单词"""
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({'error': '缺少搜索关键词'}), 400
        
        vocabulary_items = VocabularyItem.query.filter(
            VocabularyItem.word.contains(query) | 
            VocabularyItem.definition.contains(query)
        ).order_by(VocabularyItem.created_at.desc()).all()
        
        return jsonify({
            'success': True,
            'vocabulary': [item.to_dict() for item in vocabulary_items],
            'count': len(vocabulary_items)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@vocabulary_bp.route('/vocabulary/export', methods=['GET'])
def export_vocabulary():
    """导出单词本为JSON格式"""
    try:
        vocabulary_items = VocabularyItem.query.order_by(VocabularyItem.created_at.desc()).all()
        
        export_data = {
            'vocabulary': [item.to_dict() for item in vocabulary_items],
            'total_count': len(vocabulary_items),
            'export_timestamp': VocabularyItem.query.first().created_at.isoformat() if vocabulary_items else None
        }
        
        return jsonify({
            'success': True,
            'data': export_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

