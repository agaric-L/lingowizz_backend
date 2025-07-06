from flask import Blueprint, request, jsonify
from src.services.video_search import VideoSearchService

video_bp = Blueprint('video', __name__)
video_service = VideoSearchService()

@video_bp.route('/recommend', methods=['GET', 'POST'])
def recommend_videos():
    """视频推荐接口（支持 GET 和 POST）"""
    if request.method == "GET":
        user_tags = request.args.getlist("tags")
    else:  # POST
        # 支持 application/json 和 application/x-www-form-urlencoded
        if request.is_json:
            data = request.get_json()
            user_tags = data.get("tags", [])
        else:
            user_tags = request.form.getlist("tags")
    
    if not user_tags:
        return jsonify({"error": "请提供 tags 参数"}), 400

    try:
        result = video_service.get_recommended_videos(user_tags)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"获取视频推荐失败: {str(e)}"}), 500 