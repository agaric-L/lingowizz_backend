'''
环境与路径设置:
sys.path.insert(0, ...): 这是一个非常关键的技巧。它将项目的根目录（即当前文件所在目录的上一级目录）添加到了 Python 的模块搜索路径中。这使得你可以使用绝对路径导入项目内的模块，例如 from src.models.user import db，而不用担心因为运行路径不同导致 ModuleNotFoundError。
Flask 应用初始化与扩展:
app = Flask(__name__, ...): 创建一个 Flask 应用实例。static_folder 参数明确指定了前端静态文件（如 HTML, CSS, JS）的存放位置。
app.config['SECRET_KEY']: 设置一个密钥，Flask 用它来加密客户端的 session、cookie 等信息，防止被篡改。
CORS(app): 集成了 Flask-CORS 扩展，允许来自不同源（域名、协议或端口）的前端应用访问你的 API。这在前后端分离的架构中是必需的。
蓝图 (Blueprints) 注册:
app.register_blueprint(...): 这是组织大型 Flask 应用的最佳实践。代码将不同功能的路由（用户、图像处理、词汇、对话）分别定义在不同的文件（蓝图）中，然后在主程序里统一注册。
url_prefix='/api': 这个参数非常有用，它为所有注册的蓝图下的路由都自动添加了 /api 前缀。例如，user_bp 里的 /login 路由，最终访问地址会是 /api/login。
数据库集成 (SQLAlchemy):
app.config['SQLALCHEMY_DATABASE_URI']: 这是数据库的连接字符串。从内容看，你使用的是一个托管在 AWS Neon 上的 PostgreSQL 数据库，并且要求使用 SSL 加密连接 (sslmode=require)，这很安全。
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False: 禁用了 Flask-SQLAlchemy 的事件追踪系统，可以减少不必要的性能开销，是推荐的设置。
db.init_app(app): 将 SQLAlchemy 实例 db 和你的 Flask 应用 app 关联起来。
app.app_context(): 这行代码本身在这里没有实际作用。它会创建一个应用上下文然后立即销毁。可能是调试时留下的，或者是一个误解。
注释掉的 with app.app_context(): db.create_all(): 这段代码的目的是根据你在 src/models/ 中定义的模型（如 User, VocabularyItem 等），在数据库中自动创建对应的表。通常在第一次运行或需要重置数据库时手动执行。
前端静态文件服务 (Catch-all Route):
@app.route('/', ...) 和 @app.route('/<path:path>'): 这两个路由组合起来，用于服务一个单页面应用（SPA，例如用 React, Vue, Angular 构建的前端）。
工作原理:
当一个请求进来时（比如 /profile 或 /static/js/main.js）。
它首先检查请求的路径 path 是否对应 static_folder 里的一个真实存在的文件。
如果存在（如 /static/js/main.js），就使用 send_from_directory 将该文件返回。
如果不存在（如 /profile，这是一个前端路由），它就会返回 index.html。
浏览器收到 index.html 后，会加载其中的 JavaScript，然后前端的路由库会接管，解析 URL (/profile) 并显示对应的页面。
启动入口:
if __name__ == '__main__':: 这是 Python 脚本的标准入口点。
app.run(...): 启动 Flask 自带的开发服务器。
host='0.0.0.0': 让服务器监听在所有网络接口上，这样局域网内的其他设备（如手机）也可以通过你的电脑 IP 访问。
port=5000: 指定服务运行的端口。
debug=True: 开启调试模式，这会提供详细的错误页面，并在代码修改后自动重启服务器，非常适合开发阶段。
'''

import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
#from src.models.user import db
from src.models import db

from src.models.vocabulary import VocabularyItem, ConversationSession, ConversationMessage
from src.routes.user import user_bp
from src.routes.image_processing import image_bp
from src.routes.vocabulary import vocabulary_bp
from src.routes.conversation import conversation_bp
from src.routes.video import video_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# 启用CORS支持
CORS(app)

# 注册蓝图
# 注册蓝图
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(image_bp, url_prefix='/api')
app.register_blueprint(vocabulary_bp, url_prefix='/api')
app.register_blueprint(conversation_bp, url_prefix='/api')
app.register_blueprint(video_bp, url_prefix='/api')

# 数据库配置
# 使用PostgreSQL数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://fang:Fang_strong_pass123@ep-fancy-tooth-a5zb31u4-pooler.us-east-2.aws.neon.tech:5432/neondb?sslmode=require'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)# 这一行现在会正确地初始化那个唯一的 db 实例
'''
db.init_app(app)：把 Flask app 和 SQLAlchemy 绑定起来。

with app.app_context():：激活 Flask 的应用上下文，才能执行数据库操作。

db.create_all()：根据你定义的模型（model），在数据库中创建对应的表。
'''
#创建数据库中的部分表
#app.app_context()
'''
with app.app_context():
    print("--- 正在检查并创建数据库表... ---")
    db.create_all()
    print("--- 数据库表创建完成。 ---")
'''
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

