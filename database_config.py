import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL 数据库连接配置
DATABASE_CONFIG = {
    'host': "ep-fancy-tooth-a5zb31u4-pooler.us-east-2.aws.neon.tech",
    'database': "neondb",
    'user': "fang",
    'password': "Fang_strong_pass123",
    'port': "5432",
    'sslmode': "require"
}

def get_database_connection():
    """获取数据库连接"""
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        return conn
    except (Exception, psycopg2.Error) as error:
        print(f"连接数据库时出错: {error}")
        return None

def test_database_connection():
    """测试数据库连接"""
    try:
        conn = get_database_connection()
        if conn:
            cur = conn.cursor()
            print("成功连接到数据库！")
            
            # 查询 public 模式下的所有表
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            
            tables = cur.fetchall()
            print("数据库中的表 (public schema):")
            for table in tables:
                print(table[0])
            
            cur.close()
            conn.close()
            return True
    except (Exception, psycopg2.Error) as error:
        print(f"操作数据库时出错: {error}")
        return False

if __name__ == "__main__":
    test_database_connection()

