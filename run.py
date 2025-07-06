#!/usr/bin/env python3
"""
LingoWizz Backend Server
启动脚本
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.main import app

if __name__ == '__main__':
    # 从环境变量获取配置
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"🚀 Starting LingoWizz Backend Server...")
    print(f"📍 Server will run on http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📊 Database: {'Configured' if os.getenv('DATABASE_URL') else 'Not configured'}")
    print(f"🤖 OpenAI API: {'Configured' if os.getenv('OPENAI_API_KEY') else 'Not configured'}")
    print(f"🤗 Hugging Face API: {'Configured' if os.getenv('HUGGINGFACE_API_KEY') else 'Not configured'}")
    print("-" * 50)
    
    # 启动Flask应用
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

