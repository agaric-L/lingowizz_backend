from flask import current_app
import requests
import json
import time
import logging

logging.basicConfig(level=logging.DEBUG)

class APIClient:
    def __init__(self):
        self.bilibili_base_url = "https://api.bilibili.com/x/web-interface/search/all/v2"

    def search_bilibili(self, search_query):
        """调用B站搜索API，提取视频数据"""
        current_app.logger.info(f"请求B站API: {self.bilibili_base_url}, {search_query}")
        params = {
            "keyword": search_query,
            "search_type": "video",
            "page": 1,
            "pagesize": 20
        }
        current_app.logger.info(f"请求B站API: {self.bilibili_base_url}, {params}")
        try:
            time.sleep(1)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://www.bilibili.com/",
                "Origin": "https://www.bilibili.com",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9",
            }
            response = requests.get(self.bilibili_base_url, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()

            video_list = []
            # 遍历 result 数组，找到 video 类型的结果
            for item in result.get("data", {}).get("result", []):
                if item.get("result_type") == "video":
                    # 提取 video 类型结果中的 data 数组（包含具体视频信息）
                    video_list.extend(item.get("data", []))

            return video_list

        except requests.RequestException as e:
            current_app.logger.error(f"B站搜索失败: {str(e)}")
            return []
        except Exception as e:
            current_app.logger.error(f"解析B站数据失败: {str(e)}")
            return []

class PromptGenerator:
    def generate_prompt(self, user_tags):
        """生成搜索词和推荐标签"""
        # 简单的搜索词生成逻辑
        search_query = " ".join(user_tags)
        
        # 基于用户标签生成推荐标签
        recommended_tags = []
        for tag in user_tags:
            if "sport" in tag.lower() or "运动" in tag:
                recommended_tags.extend(["运动", "健身", "训练"])
            elif "music" in tag.lower() or "音乐" in tag:
                recommended_tags.extend(["音乐", "歌曲", "演奏"])
            elif "cook" in tag.lower() or "烹饪" in tag:
                recommended_tags.extend(["烹饪", "美食", "料理"])
            else:
                recommended_tags.append(tag)
        
        return {
            "search_query": search_query,
            "tags": recommended_tags[:3]  # 限制为3个标签
        }

class RankingService:
    def rank_videos(self, user_intent, videos):
        """简单的视频排序逻辑"""
        # 这里可以实现更复杂的排序算法
        # 目前只是简单返回原列表
        return videos

class VideoSearchService:
    def __init__(self):
        self.api_client = APIClient()
        self.prompt_gen = PromptGenerator()
        self.ranking_service = RankingService()

    def get_recommended_videos(self, user_tags):
        """完整流程：生成搜索词 → 搜视频 → 排序 → 返回结果"""
        # 1. 生成搜索词和标签
        prompt_data = self.prompt_gen.generate_prompt(user_tags)
        search_query = prompt_data["search_query"]
        recommended_tags = prompt_data["tags"]

        # 2. 用户意图（用于排序）
        user_intent = " ".join(user_tags)

        # 3. 搜索视频
        raw_videos = self.api_client.search_bilibili(search_query)

        videos = []
        for video in raw_videos:
            try:
                # 提取关键视频信息
                cover_url = str(video.get("pic", ""))
                if cover_url.startswith("//"):
                    cover_url = "https:" + cover_url
                elif cover_url.startswith("http:"):
                    cover_url = cover_url.replace("http:", "https:")
                video_info = {
                    "id": str(video.get("bvid", "")),
                    "title": str(video.get("title", "")).replace("<em class=\"keyword\">", "").replace("</em>", ""),
                    "cover": cover_url,
                    "url": str(f"https://www.bilibili.com/video/{video.get('bvid', '')}"),
                    "duration": str(video.get("duration", "")),
                    "play_count": int(video.get("play", 0)),
                    "up": str(video.get("author", "")),
                    "tags": (str(video.get("tag", "")) or "").split(",") + recommended_tags[:2]
                }
                videos.append(video_info)
            except Exception as e:
                current_app.logger.warning(f"解析视频失败: {str(e)}")
                continue

        # 5. 排序视频
        ranked_videos = self.ranking_service.rank_videos(user_intent, videos)

        return {
            "search_query": search_query,
            "recommended_tags": recommended_tags,
            "videos": ranked_videos
        } 