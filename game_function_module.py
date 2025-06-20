import json
import multiprocessing
import re
import subprocess
import random
import sys
import threading
import time
import tkinter as tk
import pytesseract
from PIL import Image
import numpy as np
import cv2
from airtest.core.api import *
import easyocr
from skimage.color.rgb_colors import yellow


def set_tesseract_cmd():
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
        pytesseract.pytesseract.tesseract_cmd = os.path.join(base_path, 'tesseract.exe')
    else:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

set_tesseract_cmd()


def image_to_array(image, region):
    # 检查是否传入了文件路径或者是一个图像对象
    if isinstance(image, str):  # 如果是路径字符串
        pil_image = Image.open(image)
    elif isinstance(image, Image.Image):  # 如果是已经加载的 PIL 图像对象
        pil_image = image
    else:
        raise ValueError("Invalid image input. Must be a file path or PIL Image object.")

    # 定义截图区域 [left, upper, right, lower]
    region = tuple(region)  # 确保region是一个tuple

    # 截取指定区域
    cropped_image = pil_image.crop(region)
    cropped_image.save('screenshot_cropped.png')

    # 将裁剪后的图像转换为 NumPy 数组
    image_array = np.array(cropped_image)

    # 获取图像的高度和宽度
    height, width, _ = image_array.shape

    # 创建一个二维数组，包含图像的每个像素的 (x, y) 坐标
    y_indices, x_indices = np.indices((height, width))

    # 将坐标调整为相对坐标
    relative_x = x_indices.flatten() + region[0]
    relative_y = y_indices.flatten() + region[1]

    # 将 (relative_x, relative_y) 坐标与 RGB 值合并
    result = np.column_stack((relative_x, relative_y, image_array.reshape(-1, 3)))

    # 保存为文本文件，每行包含 [x, y, R, G, B]
    with open('image_array_with_coords.txt', 'w') as f:
        for line in result:
            f.write(f"[{', '.join(map(str, line))}]\n")

    return result


# 计算 RGB 之间的欧几里得距离
def calculate_color_distance(rgb1, rgb2):
    return np.sqrt((rgb1[0] - rgb2[0]) ** 2 + (rgb1[1] - rgb2[1]) ** 2 + (rgb1[2] - rgb2[2]) ** 2)


# 在图像中遍历点并检查颜色是否匹配
def check_points_in_image(image_array, region, points, color_threshold=50):
    matched_points = []
    for point in points:
        x, y = point[0], point[1]
        if region[0] <= x <= region[2] and region[1] <= y <= region[3]:
            # 将数据转换为字典
            coordinates_to_rgb = {tuple(row[:2]): row[2:] for row in image_array}
            rgb = coordinates_to_rgb.get((x, y))
            color_distance = calculate_color_distance(rgb, point[2:])
            if color_distance < color_threshold:
                print(f"匹配成功：坐标 ({x}, {y}) 的颜色距离为 {color_distance}")
                matched_points.append((x, y))

    return matched_points


def click_button(pos, name,device):
    device.touch((pos[0], pos[1]))
    print(f'点击成功', {pos, name})


def my_click(region, points, name, color_threshold, device):
    # 检查设备是否有效
    if device is None:
        print(f"错误：设备对象为 None，无法执行点击操作")
        return False

    # 捕获屏幕截图
    try:
        # 添加重试机制
        max_retries = 3
        for retry in range(max_retries):
            try:
                screenshot = device.snapshot()
                if screenshot is None:
                    print(f"截图失败，尝试重试 {retry + 1}/{max_retries}")
                    time.sleep(1)
                    continue

                # Ensure screenshot is a PIL Image, if it's not already
                if isinstance(screenshot, np.ndarray):
                    # 转换 BGR 到 RGB
                    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                    screenshot = Image.fromarray(screenshot_rgb)
                    screenshot.save('screenshot_image.png')

                # 定义截图区域
                region = region
                # 将截图转换为数组
                start_time = time.time()
                image_array = image_to_array(screenshot, region)
                # 定义要匹配的点
                points = points
                # 查找匹配的点
                matched_points = check_points_in_image(image_array, region, points, color_threshold)
                end_time = time.time()
                # 输出匹配的点的xy坐标
                print(f"匹配的点坐标: {matched_points}")
                print(f"匹配点耗时: {end_time - start_time}秒")

                if len(matched_points) > 0:
                    rd = random.randint(0, len(matched_points) - 1)
                    click_button((matched_points[rd][0], matched_points[rd][1]), name,device)
                    return True
                else:
                    print(f"未找到匹配点，尝试重试 {retry + 1}/{max_retries}")
                    time.sleep(1)

            except Exception as e:
                print(f"截图或点击操作失败 (尝试 {retry + 1}/{max_retries}): {e}")
                time.sleep(1)

        print(f"达到最大重试次数，点击操作失败")
        return False

    except Exception as e:
        print(f"截图或点击操作失败: {e}")
        return False


def image_to_text3(region, device):
    # 捕获屏幕截图
    screenshot = device.snapshot()

    # 转换为 PIL Image 格式
    if isinstance(screenshot, np.ndarray):
        screenshot = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))

    # 裁剪区域
    cropped_image = screenshot.crop(region)

    # 转灰度
    gray = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)

    # 二值化增强对比度
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR 设置
    config = '--psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '

    start_time = time.time()
    recognized_text = pytesseract.image_to_string(thresh, config=config)
    end_time = time.time()

    print(f"识别的文字: {recognized_text.strip()}")
    print(f"OCR 耗时: {end_time - start_time:.2f} 秒")

    # 提取第一个出现的数字（连续的）
    match = (re.search(r'\d{1,3}', recognized_text))

    if match:
        number = int(match.group())
        if number>200:
            number=number//10
    else:
        number = 1  # 没有识别出数字时默认返回 1

    return number

def image_to_text(region, device):
    # 捕获屏幕截图
    screenshot = device.snapshot()

    # 转换为 PIL Image 格式
    if isinstance(screenshot, np.ndarray):
        screenshot = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))

    # 裁剪区域
    cropped_image = screenshot.crop(region)

    # 转灰度
    gray = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)

    # 可以添加二值化增强对比度（可选）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR 设置（仅数字）
    config = '--psm 7 -c tessedit_char_whitelist=0123456789'

    start_time = time.time()
    recognized_text = pytesseract.image_to_string(thresh, config=config)
    end_time = time.time()

    print(f"识别的文字: {recognized_text.strip()}")
    print(f"OCR 耗时: {end_time - start_time:.2f} 秒")

    number = recognized_text.strip().replace(',', '').replace(' ', '')
    return number if number.isdecimal() else '1'

def image_to_text2(region, device):
    # 捕获屏幕截图
    screenshot = device.snapshot()

    # 确保截图是 PIL Image 格式
    if isinstance(screenshot, np.ndarray):
        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        screenshot = Image.fromarray(screenshot_rgb)

    # 裁剪指定区域
    cropped_image = screenshot.crop(region)

    # 将裁剪的图像转换为灰度
    gray = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)

    # 使用 EasyOCR 识别文字
    reader = easyocr.Reader(['en', 'ch_sim'])  # 选择支持的语言
    start_time = time.time()
    results = reader.readtext(gray)
    end_time = time.time()

    recognized_text = ' '.join([res[1] for res in results])
    print(f"识别的文字: {recognized_text.strip()}")
    print(f"OCR 耗时: {end_time - start_time} 秒")
    number = recognized_text.strip().replace(',', '')
    if number.isdecimal():
        return number
    else:
        return '1'


def load_role_task_config(role_task_config_path):
    """
    从 role_task_config.json 读取默认配置
    如果文件不存在或解析失败，则返回一个默认字典
    """
    if not os.path.exists(role_task_config_path):
        # 文件不存在，返回一个默认结构
        return {}

    try:
        with open(role_task_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # data 应该是一个包含 "角色1" ~ "角色6" 的字典
            return data
    except Exception as e:
        print(f"读取 {role_task_config_path} 出错: {e}")
        return {}


def count_money(device_id):
    device_id = device_id.split(':')[0]+'-'+device_id.split(':')[1]
    filename = f'count_money_config/金币统计配置{device_id}.json'
    if os.path.exists(filename):
        data = load_role_task_config(filename)
        red = 0
        yellow2 = 0
        for role in data:
            for value in role.values():
                red += int(value['红币'])
                yellow2 += int(value['黄币'])
        return '红币：' + str(red) + '黄币：' + str(yellow2)
    return '暂未加载'


class GetAfkTimeController:
    def __init__(self):
        """初始化"""
        # 加载配置文件
        try:
            with open("config/副本任务配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.run_transcript_config = self.config["steps"]
                print("成功加载副本任务配置")
            with open("config/主线任务配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.main_work_config = self.config["steps"]
                print("成功加载主线任务配置")
            with open("config/星力战场配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.star_battle_config = self.config["steps"]
                print("成功加星力战场任务配置")
            with open("config/获取所有角色名称.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.get_all_role_name_config = self.config["steps"]
                print("成功加载获取所有角色名称任务配置")
            with open("config/检查邮件配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.check_mail_config = self.config["steps"]
                print("成功加载获取检查邮件任务配置")
            with open("config/清理背包配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.clear_bug_config = self.config["steps"]
                print("成功加载获取清理背包任务配置")
            with open("config/技能加点配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.skil_check_config = self.config["steps"]
                print("成功加载获取技能加点任务配置")
            with open("config/自动穿戴装备配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.check_equipment_config = self.config["steps"]
                print("成功加载获取自动穿戴装备配置")
            with open("config/领取活动奖励配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.get_activity_reward_config = self.config["steps"]
                print("成功加载获取领取活动奖励配置")
            with open("config/获取金币数量配置.json", "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.get_money_config = self.config["steps"]
                print("成功加载获取金币数量配置")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            raise

        # 加载 ADB 路径
        self.adb_path = None
        self.load_adb_path()

        # 初始化设备相关变量
        self.device_serial = []
        self.devices = {}  # 存储设备连接对象的字典

    def load_adb_path(self):
        """从配置文件加载 ADB 路径"""
        config_path = "config/adb_path.conf"
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.adb_path = f.read().strip()
                    if self.adb_path and os.path.exists(self.adb_path):
                        print(f"已加载 ADB 路径: {self.adb_path}")
                    else:
                        raise Exception("ADB 路径无效或文件不存在")
            else:
                raise Exception("未找到 ADB 路径配置文件")
        except Exception as e:
            print(f"加载 ADB 路径失败: {e}")
            raise

    def choose_device(self):
        """初始化设备连接"""
        try:
            if not self.adb_path or not os.path.exists(self.adb_path):
                raise Exception("未找到有效的 ADB 路径，请先选择 ADB 文件")

            # 使用指定的 ADB 路径
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run([self.adb_path, 'devices'], capture_output=True, text=True,startupinfo=startupinfo,creationflags=subprocess.CREATE_NO_WINDOW)
            lines = result.stdout.strip().split('\n')[1:]
            devices = []

            for line in lines:
                if line.strip():
                    serial, status = line.split()
                    if status == 'device':
                        devices.append(serial)

            if not devices:
                raise Exception("未找到已连接的设备")

            # 只处理指定的设备
            for device_id in self.device_serial:
                if device_id in devices:
                    try:
                        device = connect_device(f"android:///{device_id}")
                        self.devices[device_id] = device
                        print(f"已连接设备: {device_id}")
                    except Exception as e:
                        print(f"连接设备 {device_id} 失败: {e}")

            return self.device_serial

        except Exception as e:
            print(f"获取设备列表失败: {e}")
            raise

    def get_device(self, device_id):
        """获取指定设备ID的设备对象"""
        try:
            # 首先尝试从已连接的设备中获取
            device = self.devices.get(device_id)
            if device:
                return device

            # 如果设备未连接，尝试重新连接
            print(f"设备 {device_id} 未连接，尝试重新连接")
            try:
                device = connect_device(f"android:///{device_id}")
                self.devices[device_id] = device
                print(f"设备 {device_id} 重新连接成功")
                return device
            except Exception as e:
                print(f"设备 {device_id} 重新连接失败: {e}")
                return None

        except Exception as e:
            print(f"获取设备 {device_id} 时发生错误: {e}")
            return None

    def back_home(self, name, device_id):
        """返回主页"""
        esc_times = random.randint(3, 5)
        self.press_esc(esc_times, device_id)

        # 检查是否需要点击否按钮（有弹窗时才需要）
        for i in range(3):
            if self.find_and_click_colors(name, self.main_work_config, device_id):
                print("找到并点击否按钮")
                self.random_wait(1.0, 1.8)
                break
            else:
                print("未找到否按钮，无需处理")

    def random_delay(self, min_delay=1, max_delay=3):
        """添加随机延时，防止过快点击"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)

    def click_button(self, button_name, config, device_id):
        """点击指定名称的按钮"""
        try:
            task_info = config.get(button_name)
            if not task_info:
                print(f"未找到按钮配置: {button_name}")
                return False

            region = task_info['region']
            points = task_info['points']
            self.click_time_out(region, points, button_name, 10, self.get_device(device_id), 1)
            self.random_delay(1, 2)  # 延时
            return True
        except Exception as e:
            print(f"点击按钮失败: {e}")
            return False

    def click_button2(self, button_name, button_name2, config, device_id):
        """点击指定名称的按钮"""
        try:
            task_info = config[button_name].get(button_name2)
            if not task_info:
                print(f"未找到按钮配置: {button_name}")
                return False

            region = task_info['region']
            points = task_info['points']
            self.click_time_out(region, points, button_name2, 10, self.get_device(device_id), 1)
            self.random_delay(1, 2)  # 延时
            return True
        except Exception as e:
            print(f"点击按钮失败: {e}")
            return False

    def perform_task(self, task_name, config, device_id):
        """执行一个任务（按钮点击）"""
        for _ in range(3):
            if self.find_and_colors(task_name, config, device_id):
                if not self.click_button(task_name, config, device_id):
                    print(f"任务执行失败: {task_name}")
                return
            time.sleep(1)

    def perform_task2(self, task_name, task_name2, config, device_id):
        """执行一个任务（按钮点击）"""
        if not self.click_button2(task_name, task_name2, config, device_id):
            print(f"任务执行失败: {task_name}")

    def random_click(self, region_name, config,device_id):
        """在指定区域内随机点击"""
        try:
            if region_name not in config:
                print(f"错误：未找到区域配置 {region_name}")
                return False

            task = config[region_name]
            region = task["region"]
            print(region)
            offset = task.get("offset", 0)
            x1, y1, x2, y2 = region
            x = random.randint(x1 + offset, x2 - offset)
            y = random.randint(y1 + offset, y2 - offset)

            print(f"点击区域: {region_name}")
            print(f"区域范围: [{x1}, {y1}, {x2}, {y2}]")
            print(f"随机坐标: ({x}, {y})")

            # 安全检查
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                print(f"警告：生成的坐标 ({x}, {y}) 超出区域范围 [{x1}, {y1}, {x2}, {y2}]")
                return False

            # 使用设备特定的 touch 函数
            device = self.get_device(device_id)
            if device:
                device.touch([x, y])
                return True
            return False

        except Exception as e:
            print(f"随机点击失败: {e}")
            return False

    def random_wait(self, min_time=0.5, max_time=2.0):
        """随机等待一段时间"""
        wait_time = random.uniform(min_time, max_time)
        print(f"随机等待 {wait_time:.1f} 秒...")
        time.sleep(wait_time)

    def press_esc(self, times=2, device_id=None):
        """按ESC指定次数"""
        print(f"按ESC {times} 次")
        device = self.get_device(device_id)
        if device:
            for i in range(times):
                device.keyevent("BACK")
                time.sleep(0.1)
            self.random_wait()

    def find_and_click_colors(self, task_name, config, device_id):
        """查找并点击指定的颜色特征点"""
        try:
            task = config[task_name]
            region = task["region"]
            points = task["points"]
            tolerance = task.get("tolerance", 30)
            min_matches = task.get("min_matches", 5)

            device = self.get_device(device_id)
            if not device:
                print(f"未找到设备: {device_id}")
                return False

            screen = device.snapshot()
            if screen is None:
                return False

            screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            matched_positions = []
            print(f"\n检测 {task_name}:")
            print(f"区域范围: {region}")
            print(f"特征点数量: {len(points)}")

            for point in points:
                x, y, r, g, b = point
                try:
                    color = screen_rgb[y, x]
                    color_diff = np.sqrt(
                        (int(color[0]) - r) ** 2 +
                        (int(color[1]) - g) ** 2 +
                        (int(color[2]) - b) ** 2
                    )
                    if color_diff <= tolerance:
                        matched_positions.append((x, y))
                except IndexError:
                    continue

            if len(matched_positions) >= min_matches:
                click_pos = random.choice(matched_positions)
                print(f"随机选择点击坐标: ({click_pos[0]}, {click_pos[1]})")
                device.touch(click_pos)
                return True
            else:
                print(f"未找到 {task_name}，仅匹配到 {len(matched_positions)} 个点")
                return False

        except Exception as e:
            print(f"查找 {task_name} 时出错: {e}")
            return False
    def find_and_colors(self, task_name, config, device_id):
        """查找并点击指定的颜色特征点"""
        try:
            task = config[task_name]
            region = task["region"]
            points = task["points"]
            tolerance = task.get("tolerance", 30)
            min_matches = task.get("min_matches", 5)

            device = self.get_device(device_id)
            if not device:
                print(f"未找到设备: {device_id}")
                return False

            screen = device.snapshot()
            if screen is None:
                return False

            screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            matched_positions = []
            print(device_id)
            print(f"\n检测 {task_name}:")
            print(f"区域范围: {region}")
            print(f"特征点数量: {len(points)}")

            for point in points:
                x, y, r, g, b = point
                try:
                    color = screen_rgb[y, x]
                    color_diff = np.sqrt(
                        (int(color[0]) - r) ** 2 +
                        (int(color[1]) - g) ** 2 +
                        (int(color[2]) - b) ** 2
                    )
                    if color_diff <= tolerance:
                        matched_positions.append((x, y))
                except IndexError:
                    continue

            if len(matched_positions) >= min_matches:
                print(f'{device_id}成功找到{task_name}按钮')
                return True
            else:
                print(f"{device_id}未找到 {task_name}，仅匹配到 {len(matched_positions)} 个点")
                return False

        except Exception as e:
            print(f"查找 {task_name} 时出错: {e}")
            return False

    def click_time_out(self, region, points, name, color_threshold, device, time):
        switch = False
        while time >= 0:
            if my_click(region, points, name, color_threshold, device):
                switch = True
                print(f'{device}设备{name}点击成功')
                break
            time -= 1
            print(f'没有找到{name}按钮，尝试再次点击')
        if not switch:
            print(f'{name}点击失败')

    def get_current_grade(self,role_name):
        # 先尝试从等级配置文件中读取当前角色的目标等级
        try:
            with open("config/等级配置.json", "r", encoding="utf-8") as f:
                level_config = json.load(f)
            # 通过 role_name 获取对应的等级数值，若不存在则默认为 0
            target_grade = int(level_config.get(role_name, {}).get("等级", "20"))
        except Exception as e:
            print(f"读取等级配置文件失败: {e}")
            target_grade = 0
        return target_grade

    #冒险修炼场
    def adventure_ground(self, device_id, role_name):
        print('初始化界面')
        time.sleep(2)
        for _ in range(13):
            if self.find_and_colors('背包图标', self.get_money_config, device_id):
                self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                self.back_home('否按钮',device_id)
                break
            self.back_home('否按钮', device_id)
        self.random_wait(1, 1.8)
        self.random_click('菜单按钮', self.main_work_config, device_id)
        task_names = ['快速内容图标','冒险修炼场图标', '冒险场-略过', '冒险场-入场','冒险场-挂机时长不够确认按钮']
        for name in task_names:
            self.perform_task(name, self.main_work_config, device_id)
        time.sleep(2)
        if self.find_and_colors('开始按钮',self.star_battle_config,device_id):
            task_name2 = ['使用按钮', '开始按钮']
            for name in task_name2:
                self.perform_task(name, self.star_battle_config, device_id)
        switch = False
        for _ in range(15):
            if self.find_and_colors('退出副本按钮',self.run_transcript_config,device_id):
                switch = True
                print(f'{device_id}:进入冒险场')
                print(f'{device_id}:设备开始挂机')
                break
            time.sleep(0.5)
        if switch:
            max_wait = 7200  # 最长挂机时间（秒）
            interval = 10  # 每次检查间隔（秒）
            elapsed = 0

            while elapsed < max_wait:
                print(f'{device_id}: 正在挂机中，已过 {elapsed} 秒')
                time.sleep(interval)
                elapsed += interval

                if self.find_and_click_colors('冒险岛结束广告按钮',self.main_work_config,device_id):
                    time.sleep(10)
                    break

                if (self.find_and_click_colors('冒险场离开按钮',self.main_work_config,device_id) and
                        self.find_and_colors('冒险岛一百级字眼', self.main_work_config, device_id)):
                    time.sleep(10)
                    break

                if not self.find_and_colors('冒险岛一百级字眼',self.main_work_config,device_id):
                    self.find_and_click_colors('冒险场确认按钮',self.main_work_config,device_id)

                self.find_and_click_colors('自动分配按钮', self.main_work_config, device_id)

                if self.find_and_click_colors('配戴按钮', self.main_work_config, device_id):
                    self.find_and_click_colors('确认配戴按钮', self.main_work_config, device_id)

                time.sleep(10)
                # if current_grade >= 100:
                #     break
                #一百级冒险修炼场就结束了
            print(f'{device_id}: 挂机结束')
        else:
            print('没有进入冒险场')
            self.adventure_ground(device_id, role_name)


    def work1(self, device_id, role_name):
        """执行领取挂机时间流程"""
        try:
            # 1. 初始化界面
            print("\n步骤1: 初始化游戏界面")
            self.back_home('否按钮', device_id)
            self.random_wait(1, 1.8)
            # 3. 点击活动按钮
            self.random_click('菜单按钮', self.main_work_config,device_id)
            task_names = ['任务图标', '开始任务', '对话框跳过按钮1', '对话框确认跳过按钮']
            for name in task_names:
                    self.perform_task(name, self.main_work_config, device_id)
            print("\n步骤1: 初始化游戏界面")
            # 再次检查是否需要点击否按钮
            # self.back_home('否按钮', device_id)
        except Exception as e:
            print(f"主线任务失败: {e}")

    def work2(self,device_id,role_name):
            try:
                # current_grade = 0
                # self.find_and_click_colors('否按钮', self.main_work_config, device_id)
                #
                # self.find_and_click_colors('关闭广告按钮', self.main_work_config, device_id)

                if self.find_and_colors('背包图标', self.get_money_config, device_id):
                    time.sleep(0.5)
                    current_grade = int(image_to_text([36, 11, 177, 32], self.get_device(device_id)))
                    print(current_grade)
                #     time.sleep(0.5)
                # self.find_and_click_colors("对话框跳过按钮1", self.main_work_config, device_id)
                # self.random_wait(1,1.8)
                # self.find_and_click_colors("对话框确认跳过按钮", self.main_work_config, device_id)
                # self.random_wait(1, 1.8)
                #
                # self.find_and_click_colors("可开始按钮", self.main_work_config, device_id)
                #
                # self.find_and_click_colors("可开始按钮2", self.main_work_config, device_id)
                #
                # self.random_wait(1, 1.8)
                #
                #
                # if self.find_and_click_colors("领取奖励", self.main_work_config, device_id):
                #     print("找到并点击领取奖励")
                #     return True, current_grade
                # return False,current_grade
            except Exception as e:
                print(f'主线任务部分出错，{e}')

    # 主线任务-带有冒险修炼场
    def main_work(self, device_id, role_name):
        """执行领取挂机时间流程"""
        try:
            # 先尝试从等级配置文件中读取当前角色的目标等级
            try:
                with open("config/等级配置.json", "r", encoding="utf-8") as f:
                    level_config = json.load(f)
                # 通过 role_name 获取对应的等级数值，若不存在则默认为 0
                target_grade = int(level_config.get(role_name, {}).get("等级", "20"))
            except Exception as e:
                print(f"读取等级配置文件失败: {e}")
                target_grade = 0
            pass_count = 0
            required_confirmations = 2  # 连续几次确认后才认为等级达标
            current_grade = 0
            while True:
                # 1. 初始化界面
                print("\n步骤1: 初始化游戏界面")
                for _ in range(5):
                    if self.find_and_colors('背包图标', self.get_money_config, device_id):
                        break
                    self.back_home('否按钮', device_id)
                # 获取当前等级，通过 OCR 识别
                if self.find_and_colors('背包图标', self.get_money_config, device_id):
                    #60, 11, 88, 28
                    time.sleep(0.5)
                    current_grade = int(image_to_text3([36, 11, 177, 32], self.get_device(device_id)))
                    # log_text.insert(tk.END, f"设备{device_id}: 当前等级{current_grade}\n")
                    time.sleep(0.5)
                    # 用目标等级替代原有的 20 进行判断
                    print(device_id,'当前等级:',current_grade)
                    print(target_grade, '限制等级')
                    if current_grade >= 200:
                        current_grade = 19
                    if current_grade >= target_grade:
                        pass_count += 1
                        print(
                            f"{device_id}当前等级为{current_grade}，已经达到或超出目标等级{target_grade},pass_count:{pass_count} 继续重试")
                        if pass_count >= required_confirmations:
                            print(f"{device_id}当前等级为{current_grade}，已经达到或超出目标等级{target_grade}，无需执行任务")
                            if current_grade<100 and not 30<current_grade<45:
                                self.adventure_ground(device_id, role_name)
                                print(f'{device_id}:结束主线任务')
                                break
                            elif current_grade<100:
                                pass
                            else:
                                print('当前等级大于100，不执行冒险修炼场')
                                break

                self.random_wait(1, 1.8)
                # 3. 点击活动按钮
                if self.find_and_colors('书卷图标', self.main_work_config, device_id):
                    self.random_click('菜单按钮', self.main_work_config, device_id)
                    task_names = ['任务图标', '开始任务', '对话框跳过按钮1', '对话框确认跳过按钮']

                    for name in task_names:
                        self.perform_task(name, self.main_work_config, device_id)

                    if self.find_and_colors('小企鹅图标', self.main_work_config, device_id):
                        self.back_home('否按钮',device_id)
                        break
                else:
                    print('还在执行任务，不需要重新点击任务')
                temp = 0
                no_action_count = 0  # 连续无点击次数统计
                max_no_action = 5  # 超过这个次数就退出

                while True:
                    action_triggered = False  # 当前轮是否有点击成功

                    if self.find_and_click_colors('放弃按钮', self.main_work_config, device_id):
                        self.find_and_click_colors('放弃确定按钮', self.main_work_config, device_id)
                        time.sleep(3)
                        self.random_click('自动战斗按钮2', self.star_battle_config, device_id)

                    if self.find_and_click_colors("新内容开放", self.star_battle_config, device_id):
                        if current_grade>30:
                            time.sleep(10)
                            self.adventure_ground(device_id, role_name)

                    if self.find_and_click_colors('否按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('任务移动按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors("广告2", self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors("广告3", self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('放弃按钮', self.main_work_config, device_id):
                        self.find_and_click_colors('放弃确定按钮', self.main_work_config, device_id)
                        time.sleep(3)
                        self.random_click('自动战斗按钮2', self.star_battle_config, device_id)

                    if self.find_and_click_colors('所有任务执行完确认按钮', self.main_work_config, device_id):
                        if self.find_and_click_colors('所有任务执行完确认按钮', self.main_work_config, device_id):
                            if self.find_and_click_colors('所有任务执行完确认按钮', self.main_work_config, device_id):
                                pass
                        break  # 所有任务执行完确认结束

                    if self.find_and_click_colors('否按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('配戴按钮', self.main_work_config, device_id):
                        self.find_and_click_colors('确认配戴按钮', self.main_work_config, device_id)

                    self.find_and_click_colors('自动分配按钮', self.main_work_config, device_id)

                    if self.find_and_click_colors("新内容开放", self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('放弃按钮', self.main_work_config, device_id):
                        self.find_and_click_colors('放弃确定按钮', self.main_work_config, device_id)
                        time.sleep(3)
                        self.random_click('自动战斗按钮2', self.star_battle_config, device_id)

                    if self.find_and_click_colors('村庄复活按钮', self.main_work_config, device_id):
                          break

                    if self.find_and_click_colors('所有任务执行完确认按钮', self.main_work_config, device_id):
                        if self.find_and_click_colors('所有任务执行完确认按钮', self.main_work_config, device_id):
                            if self.find_and_click_colors('所有任务执行完确认按钮', self.main_work_config, device_id):
                                pass
                        break  # 所有任务执行完确认结束

                    if self.find_and_click_colors('广告关闭按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('放弃按钮', self.main_work_config, device_id):
                        self.find_and_click_colors('放弃确定按钮', self.main_work_config, device_id)
                        time.sleep(3)
                        self.random_click('自动战斗按钮2', self.star_battle_config, device_id)


                    if self.find_and_click_colors('所有任务执行完确认按钮2', self.main_work_config, device_id):
                        if self.find_and_click_colors('所有任务执行完确认按钮2', self.main_work_config, device_id):
                            if self.find_and_click_colors('所有任务执行完确认按钮2', self.main_work_config, device_id):
                                pass
                        break  # 所有任务执行完确认结束

                    if self.find_and_click_colors('关闭广告按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('放弃按钮', self.main_work_config, device_id):
                        self.find_and_click_colors('放弃确定按钮', self.main_work_config, device_id)
                        time.sleep(3)
                        self.random_click('自动战斗按钮2', self.star_battle_config, device_id)

                    if self.find_and_colors('任务图标', self.main_work_config, device_id):
                        time.sleep(1)
                        print(device_id, '误触了')
                        self.press_esc(1, device_id)

                    if self.find_and_click_colors("对话框跳过按钮1", self.main_work_config, device_id):
                        if self.find_and_colors('任务图标', self.main_work_config, device_id):
                            time.sleep(1)
                            print(device_id, '误触了')
                            self.press_esc(1, device_id)
                        action_triggered = True
                        self.random_wait(1, 1.8)

                    if self.find_and_click_colors("对话框确认跳过按钮", self.main_work_config, device_id):
                        action_triggered = True
                        self.random_wait(1, 1.8)

                    if self.find_and_click_colors("可开始按钮", self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors("广告1", self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors("可开始按钮2", self.main_work_config, device_id):
                        action_triggered = True
                        self.random_wait(1, 1.8)

                    if self.find_and_click_colors("领取奖励", self.main_work_config, device_id):
                        if self.find_and_click_colors("领取奖励", self.main_work_config, device_id):
                            if self.find_and_click_colors("领取奖励", self.main_work_config, device_id):
                                pass
                        break  # 正常结束


                    temp += 1

                    if action_triggered:
                        no_action_count = 0
                    else:
                        no_action_count += 1
                        print(f"{device_id}: 连续未触发点击次数 {no_action_count}")

                    print(device_id, ':设备超时时间', temp)
                    if temp > 9:
                        print(device_id, ": 超时执行达到最大次数，退出循环")
                        break

                    if no_action_count >= max_no_action:
                        print(device_id, ": 长时间无点击响应，主动退出循环")
                        break

            print("\n步骤1: 初始化游戏界面")
        except Exception as e:
            print(f"主线任务失败: {e}")


    # 主线任务-没有冒险修炼场
    def main_work2(self, device_id, role_name):
        """执行领取挂机时间流程"""
        try:
            # 先尝试从等级配置文件中读取当前角色的目标等级
            try:
                with open("config/等级配置.json", "r", encoding="utf-8") as f:
                    level_config = json.load(f)
                # 通过 role_name 获取对应的等级数值，若不存在则默认为 0
                target_grade = int(level_config.get(role_name, {}).get("等级", "20"))
            except Exception as e:
                print(f"读取等级配置文件失败: {e}")
                target_grade = 0
            pass_count = 0
            required_confirmations = 3  # 连续几次确认后才认为等级达标
            while True:
                # 1. 初始化界面
                print("\n步骤1: 初始化游戏界面")
                self.back_home('否按钮', device_id)
                # 获取当前等级，通过 OCR 识别
                if self.find_and_colors('背包图标', self.get_money_config, device_id):
                    time.sleep(0.5)
                    current_grade = int(image_to_text([60, 11, 88, 28], self.get_device(device_id)))
                    # 用目标等级替代原有的 20 进行判断
                    print(target_grade, '限制等级')
                    if current_grade>=1000:
                        current_grade=1
                    if current_grade >= target_grade:
                        pass_count+=1
                        print(f"当前等级为{current_grade}，已经达到或超出目标等级{target_grade},pass_count:{pass_count} 继续重试")
                        if pass_count >= required_confirmations:
                            print(f"当前等级为{current_grade}，已经达到或超出目标等级{target_grade}，无需执行任务")
                            break
                self.random_wait(1, 1.8)
                # 3. 点击活动按钮
                self.random_click('菜单按钮', self.main_work_config,device_id)
                task_names = ['任务图标', '开始任务', '对话框跳过按钮1', '对话框确认跳过按钮']
                for name in task_names:
                    self.perform_task(name, self.main_work_config, device_id)

                temp = 0
                no_action_count = 0  # 连续无点击次数统计
                max_no_action = 20  # 超过这个次数就退出

                while True:
                    action_triggered = False  # 当前轮是否有点击成功

                    if self.find_and_click_colors('否按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('所有任务执行完确认按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('任务移动按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors('关闭广告按钮', self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors("对话框跳过按钮1", self.main_work_config, device_id):
                        if self.find_and_colors('任务图标',self.main_work_config,device_id):
                            time.sleep(1)
                            print(device_id,'误触了')
                            self.press_esc(1,device_id)
                        action_triggered = True
                    self.random_wait(1, 1.8)

                    if self.find_and_click_colors("对话框确认跳过按钮", self.main_work_config, device_id):
                        action_triggered = True
                    self.random_wait(1, 1.8)

                    if self.find_and_click_colors("可开始按钮", self.main_work_config, device_id):
                        action_triggered = True

                    if self.find_and_click_colors("可开始按钮2", self.main_work_config, device_id):
                        action_triggered = True

                    self.random_wait(1, 1.8)

                    if self.find_and_click_colors("领取奖励", self.main_work_config, device_id):
                        break  # 正常结束

                    if self.find_and_click_colors('放弃按钮', self.main_work_config, device_id):
                        self.find_and_click_colors('放弃确定按钮', self.main_work_config, device_id)
                        break  # 放弃任务结束

                    temp += 1

                    if action_triggered:
                        no_action_count = 0
                    else:
                        no_action_count += 1
                        print(f"{device_id}: 连续未触发点击次数 {no_action_count}")

                    print(device_id, ':设备超时时间', temp)
                    if temp > 50:
                        print(device_id, ": 超时执行达到最大次数，退出循环")
                        break

                    if no_action_count >= max_no_action:
                        print(device_id, ": 长时间无点击响应，主动退出循环")
                        break


            print("\n步骤1: 初始化游戏界面")
        except Exception as e:
            print(f"主线任务失败: {e}")

    # 精英地城
    def elite_dungeons(self, fail_count=0, device_id=None):
        """执行精英地城任务"""
        device = self.get_device(device_id)
        if not device:
            print(f"未找到设备: {device_id}")
            return
        for _ in range(13):
            if self.find_and_colors('背包图标', self.get_money_config, device_id):
                self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                time.sleep(1)
                self.back_home('否按钮', device_id)
                break
            time.sleep(1)
            self.back_home('否按钮', device_id)

        self.random_click('菜单按钮', self.run_transcript_config,device_id)
        task_name = ['快速内容图标', '精英地城', '略过按钮','快速组队按钮', '确认进入按钮']
        for name in task_name:
            self.perform_task2('精英地城', name, self.run_transcript_config, device_id)

        self.random_wait(7, 9)

        for _ in range(60):
            if not self.find_and_colors("机器人图标", self.run_transcript_config, device_id):
                break
            time.sleep(1)
            print('检测到机器人，继续等待')

        temp = False
        time.sleep(10)
        for _ in range(7):
            if self.find_and_colors("退出副本按钮", self.run_transcript_config, device_id):
                temp = True
                break
            time.sleep(1)

        if not temp:
            print('没有找到退出副本按钮,重新进入')
            fail_count += 1
            if fail_count >= 2:
                print("连续3次未找到退出副本按钮，程序结束")
                return
            else:
                self.elite_dungeons(fail_count, device_id)
        else:
            print('已经进入副本')
            max_time = 300
            min = 0
            while True:
                time.sleep(5)
                min+=5
                if (self.find_and_click_colors("离开按钮-精英地城", self.run_transcript_config, device_id)
                        or self.find_and_click_colors('精英地城失败离开按钮',self.run_transcript_config,device_id)):
                    break

                if max_time<= min:
                    break


    # 每日地城
    def everyday_dungeons(self, fail_count=0, device_id=None):
        """执行每日地城任务"""
        device = self.get_device(device_id)
        if not device:
            print(f"未找到设备: {device_id}")
            return

        for _ in range(13):
            if self.find_and_colors('背包图标', self.get_money_config, device_id):
                self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                time.sleep(1)
                self.back_home('否按钮', device_id)
                break
            time.sleep(1)
            self.back_home('否按钮', device_id)
        # 执行每日地城
        self.random_click('菜单按钮', self.run_transcript_config,device_id)
        task_name2 = ['快速内容图标', '每日地城','略过按钮', '进入按钮', '每日确认按钮']
        for name in task_name2:
            self.perform_task2('每日地城', name, self.run_transcript_config, device_id)

        self.random_wait(7, 9)

        for _ in range(60):
            if not self.find_and_colors("机器人图标", self.run_transcript_config, device_id):
                break
            time.sleep(1)
            print('检测到机器人，继续等待')

        temp = False
        time.sleep(10)
        for _ in range(7):
            if self.find_and_colors("退出副本按钮", self.run_transcript_config, device_id):
                temp = True
                break
            time.sleep(1)

        if not temp:
            print('没有找到退出副本按钮,重新进入')
            fail_count += 1
            if fail_count >= 2:
                print("连续3次未找到退出副本按钮，程序结束")
                return
            else:
                self.everyday_dungeons(fail_count, device_id)
        else:
            print('已经进入副本')
            max_time = 300
            min_time = 0
            while True:
                if (self.find_and_click_colors("每日离开按钮", self.run_transcript_config, device_id) or
                        self.find_and_click_colors("每日地城-阵亡离开按钮", self.run_transcript_config, device_id)):
                    break
                if max_time<= min_time:
                    break
                min_time+=5
                time.sleep(5)

    # 金字塔
    def pyramid(self, fail_count=0, device_id=None):
        """执行奈特的金字塔任务"""
        device = self.get_device(device_id)
        if not device:
            print(f"未找到设备: {device_id}")
            return
        # 执行奈特的金字塔
        for _ in range(13):
            if self.find_and_colors('背包图标', self.get_money_config, device_id):
                self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                time.sleep(1)
                self.back_home('否按钮', device_id)
                break
            time.sleep(1)
            self.back_home('否按钮', device_id)
        self.random_click('菜单按钮', self.run_transcript_config,device_id)
        task_name2 = ['快速内容图标', '奈特的金字塔', '略过按钮','快速组队按钮', '金字塔确认按钮']
        for name in task_name2:
            self.perform_task2('奈特的金字塔', name, self.run_transcript_config, device_id)

        self.random_wait(7, 9)

        for _ in range(60):
            if not self.find_and_colors("机器人图标", self.run_transcript_config, device_id):
                break
            time.sleep(1)
            print('检测到机器人，继续等待')

        temp = False
        time.sleep(10)
        for _ in range(7):
            if self.find_and_colors("退出副本按钮", self.run_transcript_config, device_id):
                temp = True
                break
            time.sleep(1)

        if not temp:
            print('没有找到退出副本按钮,重新进入')
            fail_count += 1
            if fail_count >= 2:
                print("连续3次未找到退出副本按钮，程序结束")
                return
            else:
                self.pyramid(fail_count, device_id)
        else:
            print('已经进入副本')
            max_time = 350
            min_time = 0
            while True:
                if self.find_and_click_colors("金字塔离开按钮", self.run_transcript_config, device_id):
                    break
                if min_time>=max_time:
                    break
                min_time+=5
                time.sleep(5)

    # 一键执行副本
    def run_transcript(self, device_id, role_name=None):
        try:
            # 1. 初始化界面
            print("\n步骤1: 初始化游戏界面")
            for _ in range(13):
                if self.find_and_colors('背包图标', self.get_money_config, device_id):
                    break
                self.back_home('否按钮', device_id)
            self.elite_dungeons(device_id=device_id)
            self.random_wait(2,3)
            self.everyday_dungeons(device_id=device_id)
            self.random_wait(2, 3)
            self.pyramid(device_id=device_id)

        except Exception as e:
            print(f"副本任务执行失败: {e}")

    # 执行星力战场
    def run_star_battle(self, device_id,role_name=None,fail_count=0,timeout=0):

        try:
            # 检查设备连接状态
            device = self.get_device(device_id)
            if device is None:
                print(f"无法获取设备 {device_id}，星力战场任务终止")
                return

            time.sleep(2)
            # 1. 初始化界面
            print("\n步骤1: 初始化游戏界面")
            time.sleep(1)
            # 添加重试机制
            max_retries = 3
            for retry in range(max_retries):
                try:
                    for _ in range(13):
                        if self.find_and_colors('背包图标', self.get_money_config, device_id):
                            self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                            self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                            time.sleep(1)
                            self.back_home('否按钮', device_id)
                            break
                        time.sleep(1)
                        self.back_home('否按钮', device_id)
                    self.buy_medicine(device_id)
                    self.random_click('菜单按钮', self.star_battle_config,device_id)
                    task_name = ['快速内容图标', '星力战场图标', '适当按钮']

                    for name in task_name:
                        # 每次点击前检查设备状态
                        device = self.get_device(device_id)
                        if device is None:
                            print(f"设备 {device_id} 连接断开，尝试重新连接")
                            continue

                        self.perform_task(name, self.star_battle_config, device_id)
                        time.sleep(1)  # 添加短暂延时
                    self.random_click('选择队伍', self.star_battle_config,device_id)

                    task_name2 = ['申请加入', '确认']
                    for name in task_name2:
                        # 每次点击前检查设备状态
                        device = self.get_device(device_id)
                        if device is None:
                            print(f"设备 {device_id} 连接断开，尝试重新连接")
                            continue

                        self.perform_task(name, self.star_battle_config, device_id)
                        time.sleep(1)  # 添加短暂延时

                    number1 = 0
                    for _ in range(30):
                        if self.find_and_colors("快速内容图标", self.star_battle_config, device_id):
                            number1 += 1
                        if number1 >= 3:
                            break
                        time.sleep(1.5)

                    if number1 >= 3:
                        break

                except Exception as e:
                    print(f"星力战场任务执行失败 (尝试 {retry + 1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        time.sleep(2)  # 重试前等待
                        continue
                    else:
                        print("达到最大重试次数，星力战场任务终止")
                        return

            self.random_wait(2, 3.5)

            self.back_home('否按钮', device_id)
            time.sleep(1)
            self.random_click('自动战斗按钮', self.star_battle_config, device_id)
            self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
            task_name2 = ['使用按钮', '开始按钮']
            for name in task_name2:
                device = self.get_device(device_id)
                if device is None:
                    print(f"设备 {device_id} 连接断开，无法继续执行")
                    return
                self.perform_task(name, self.star_battle_config, device_id)
                time.sleep(1)
            if self.find_and_colors("开始按钮", self.star_battle_config, device_id):
                self.find_and_click_colors("开始按钮", self.star_battle_config, device_id)
            self.random_wait(3, 2.5)
            temp = False
            for _ in range(18):
                if (self.find_and_colors("战场人数图标", self.star_battle_config, device_id) or
                        self.find_and_colors("血条图标", self.star_battle_config, device_id)):
                    # self.press_esc(1, device_id)
                    temp = True
                    break
                time.sleep(1)

            if not temp:
                print('没有找到战场人数图标,重新进入')
                fail_count += 1
                if fail_count >= 3:
                    print("连续3次未找到血条，程序结束")
                    return
                else:
                    self.run_star_battle(device_id=device_id,fail_count=fail_count)
            else:
                print('已经进入星力战场')
                while True:
                    if self.find_and_click_colors("村庄复活", self.star_battle_config, device_id):
                        time.sleep(3)
                        self.back_home('否按钮', device_id)
                        time.sleep(3)
                        self.find_and_click_colors('自动加入按钮', self.star_battle_config, device_id)
                        time.sleep(1)
                        self.find_and_click_colors('自动加入确认按钮', self.star_battle_config, device_id)
                        for _ in range(30):
                            if self.find_and_colors("战场人数图标", self.star_battle_config, device_id):
                                print('阵亡后已经进入星力战场')
                                break
                        self.random_click('自动战斗按钮', self.star_battle_config, device_id)
                        self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                        task_name2 = ['使用按钮', '开始按钮']
                        for name in task_name2:
                            device = self.get_device(device_id)
                            if device is None:
                                print(f"设备 {device_id} 连接断开，无法继续执行")
                                return
                            self.perform_task(name, self.star_battle_config, device_id)
                            time.sleep(1)

                    if self.find_and_click_colors('自动加入按钮', self.star_battle_config, device_id):
                        time.sleep(2)
                        self.find_and_click_colors('自动加入确认按钮', self.star_battle_config, device_id)
                        for _ in range(30):
                            if self.find_and_colors("战场人数图标", self.star_battle_config, device_id):
                                print('阵亡后已经进入星力战场')
                                break
                        self.random_click('自动战斗按钮', self.star_battle_config, device_id)
                        self.random_click('自动战斗按钮2', self.star_battle_config, device_id)
                        task_name2 = ['使用按钮', '开始按钮']
                        for name in task_name2:
                            device = self.get_device(device_id)
                            if device is None:
                                print(f"设备 {device_id} 连接断开，无法继续执行")
                                return
                            self.perform_task(name, self.star_battle_config, device_id)
                            time.sleep(1)

                    if self.find_and_click_colors('配戴按钮', self.main_work_config, device_id):
                        self.find_and_click_colors('确认配戴按钮', self.main_work_config, device_id)

                    self.find_and_click_colors("新内容开放", self.star_battle_config, device_id)

                    if timeout>7200:
                        print('超时退出')
                        break
                    timeout += 7
                    time.sleep(7)


        except Exception as e:
            print(f'运行星力战场失败: {e}')

    # 获取所有角色名称角色名称
    def get_all_role_name(self, device_id):
        self.back_home('否按钮', device_id)
        self.random_click('菜单按钮', self.get_all_role_name_config,device_id)
        self.random_wait(1, 1.5)
        self.click_time_out(self.get_all_role_name_config['角色切换按钮']['region'],
                            self.get_all_role_name_config['角色切换按钮']['points'],
                            '角色切换按钮', 10,
                            self.get_device(device_id), 5)
        self.random_wait(1, 1.5)
        name_list = [image_to_text(self.get_all_role_name_config['当前角色']['region'], self.get_device(device_id))]
        time.sleep(1)
        self.back_home('否按钮', device_id)
        self.press_esc(1, device_id)
        self.random_wait(1, 1.8)
        self.click_time_out(self.get_all_role_name_config['角色选择页面']['region'],
                            self.get_all_role_name_config['角色选择页面']['points'],
                            '角色选择页面', 10,
                            self.get_device(device_id), 5)

        self.random_wait(7, 8)

        for i in range(1, 7):
            number = f'角色{i}'
            name_list.append(image_to_text(self.get_all_role_name_config[number]['region'], self.get_device(device_id)))
        name_list.append(device_id)

        return name_list

    def get_money(self, device_id, role_name):
        """执行领取挂机时间流程"""
        try:
            # 1. 初始化界面
            print("\n步骤1: 初始化游戏界面")
            self.back_home('否按钮', device_id)

            # self.random_wait(1, 1.8)
            # 3. 点击活动按钮
            self.click_time_out(self.get_money_config['背包图标']['region'],
                                self.get_money_config['背包图标']['points'], '背包图标', 10, self.get_device(device_id),
                                5)
            self.random_wait(3, 4)
            self.random_click('金币图标', self.get_money_config,device_id)
            self.random_click('金币图标2', self.get_money_config, device_id)
            # self.click_time_out(self.get_money_config['金币图标']['region'], self.get_money_config['金币图标']['points'], '金币图标', 10, self.get_device(device_id), 5)
            self.random_wait(2, 2.8)
            # 获取金币数量
            money = image_to_text([689, 367, 887, 412], self.get_device(device_id))
            print(f"金币数量: {money}")
            time.sleep(2)
            money2 = image_to_text([396, 369, 584, 411], self.get_device(device_id))
            print(f'金币数量2:{money2}')
            self.random_wait(1.2, 1.8)
            name = device_id.split(':')[0]+'-'+device_id.split(':')[1]
            new_content = {role_name: {'黄币': money, '红币': money2}}

            if money and money2:
                filename = f'count_money_config/金币统计配置{name}.json'
                print(filename)

                # 初始化数据列表
                data = []

                # 如果文件存在，读取已有数据
                if os.path.exists(filename):
                    try:
                        with open(filename, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        data = []

                # 检查角色是否存在并更新/添加数据
                found = False
                for item in data:
                    if role_name in item:
                        item[role_name]['黄币'] = money
                        item[role_name]['红币'] = money2
                        found = True
                        break

                # 如果角色不存在则添加新条目
                if not found:
                    data.append(new_content)

                # 写回文件
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            print("\n步骤1: 初始化游戏界面")
            self.back_home('否按钮', device_id)

        except Exception as e:
            print(f"获取金币失败: {e}")

    # 切换角色
    def switch_role(self, name, device_id):
        self.back_home('否按钮', device_id)

        self.press_esc(1, device_id)
        self.click_time_out(self.get_all_role_name_config['角色选择页面']['region'],
                            self.get_all_role_name_config['角色选择页面']['points'],
                            '角色选择页面', 10, self.get_device(device_id), 5)

        for _ in range(120):
            if self.find_and_colors('游戏开始',self.get_all_role_name_config,device_id):
                break
            time.sleep(1)
        print('已经来到角色面板')
        for _ in range(3):
            self.random_click(name, self.get_all_role_name_config,device_id)
            time.sleep(1.5)
        self.random_wait(1, 1.5)
        self.click_time_out(self.get_all_role_name_config['游戏开始']['region'],
                            self.get_all_role_name_config['游戏开始']['points'],
                            '游戏开始', 10, self.get_device(device_id), 5)
        for _ in range(120):
            if self.find_and_colors('背包图标', self.get_money_config, device_id):
                break
            time.sleep(1)
            self.back_home('否按钮', device_id)
        time.sleep(2)
        self.get_money(device_id, name)

    # 获取挂机时间
    def get_afk_time(self, device_id):
        print('初始化界面')
        self.back_home('否按钮', device_id)
        self.random_click('自动战斗按钮', self.star_battle_config, device_id)
        self.random_click('自动战斗按钮3', self.star_battle_config, device_id)
        time.sleep(1)
        task_name2 = ['使用按钮', '关闭按钮']
        for name in task_name2:
            self.perform_task(name, self.star_battle_config, device_id)

        time.sleep(2)

    # 检查邮箱
    def check_mail(self, device_id):
        print('初始化界面')
        self.back_home('否按钮', device_id)
        task_name2 = ['邮件图标', '全部领取', '确认', '个人按钮', '全部领取', '确认', '关闭按钮']
        for name in task_name2:
            self.perform_task(name, self.check_mail_config, device_id)
        time.sleep(2)

    # 清理背包
    def clear_bug(self, device_id):
        print('初始化界面')
        self.back_home('否按钮', device_id)
        task_name2 = ['背包图标', '蓝色售卖按钮', '设置按钮', '类型-全部', '类型-武器', '类型-防具', '类型-饰品'
            , '等级-全部', '等级-普通', '等级-稀有', '等级-史诗', '潜在能力-无','潜在能力-稀有','潜在能力-史诗' '药水按钮', '关闭贩卖药水', '套用',
                      '黄色贩卖','黄色贩卖2']
        for name in task_name2:
            self.perform_task(name, self.clear_bug_config, device_id)
            time.sleep(0.5)
        self.press_esc(2, device_id)
        time.sleep(2)

    # 添加技能的辅助函数
    def skil_proces(self, number=0, device_id=None):

        self.random_wait(1, 1.5)
        if self.find_and_colors('装备1', self.skil_check_config,device_id):
            self.click_time_out(self.skil_check_config['装备1']['region'], self.skil_check_config['装备1']['points'],
                            '装备1', 10, self.get_device(device_id), 5)
            if number >= 5:
                number = 0
            number += 1
            print(number)
            self.random_wait(1, 1.5)
            self.random_click(f'添加{number}', self.skil_check_config,device_id)
        self.random_wait(1, 1.5)

        if self.find_and_colors('装备2', self.skil_check_config, device_id):
            self.click_time_out(self.skil_check_config['装备2']['region'], self.skil_check_config['装备2']['points'],
                            '装备2', 10, self.get_device(device_id), 5)
            if number >= 5:
                number = 0
            number += 1
            print(number)
            self.random_wait(1, 1.5)
            self.random_click(f'添加{number}', self.skil_check_config,device_id)
        self.random_wait(1, 1.5)
        if self.find_and_colors('装备3', self.skil_check_config, device_id):
            self.click_time_out(self.skil_check_config['装备3']['region'], self.skil_check_config['装备3']['points'],
                            '装备3', 10, self.get_device(device_id), 5)
            if number >= 5:
                number = 0
            number += 1
            print(number)
            self.random_wait(1, 1.5)
            self.random_click(f'添加{number}', self.skil_check_config,device_id)
        self.random_wait(1, 1.5)
        if self.find_and_colors('装备4', self.skil_check_config, device_id):
            self.click_time_out(self.skil_check_config['装备4']['region'], self.skil_check_config['装备4']['points'],
                            '装备4', 10, self.get_device(device_id), 5)
            if number >= 5:
                number = 0
            number += 1
            print(number)
            self.random_wait(1, 1.5)
            self.random_click(f'添加{number}', self.skil_check_config,device_id)
        self.random_wait(1, 1.5)
        return number

    # 添加技能
    def skil_check(self, device_id):
        print('初始化界面')
        self.back_home('否按钮', device_id)
        self.random_click('菜单按钮', self.skil_check_config,device_id)
        self.random_wait(1, 1.5)
        self.click_time_out(self.skil_check_config['技能图标']['region'],
                            self.skil_check_config['技能图标']['points'],
                            '技能图标', 10,
                            self.get_device(device_id), 5)
        self.random_wait(1, 1.5)
        self.random_click('30级', self.skil_check_config,device_id)
        number = self.skil_proces(0, device_id)
        self.random_click('60级', self.skil_check_config,device_id)
        number = self.skil_proces(number, device_id)
        self.random_click('100级', self.skil_check_config,device_id)
        self.skil_proces(number, device_id)

        self.press_esc(2, device_id)
        time.sleep(2)

    # 更新装备的辅助函数
    def equipment_proces(self, device_id):
        time.sleep(0.5)
        number1 = image_to_text(self.check_equipment_config['原先装备战力区域']['region'], self.get_device(device_id))
        time.sleep(0.5)
        number2 = image_to_text(self.check_equipment_config['新装备战力区域']['region'], self.get_device(device_id))
        time.sleep(0.5)
        print('原先', number1, '新', number2)
        if number1.isdecimal():
            print('number1是数字')
            if int(number1) < int(number2):
                print('新装备厉害')
                self.click_time_out(self.check_equipment_config['装备按钮']['region'],
                                    self.check_equipment_config['装备按钮']['points'],
                                    '装备按钮', 10,
                                    self.get_device(device_id), 5)
            else:
                print('新装备不厉害')
                self.click_time_out(self.check_equipment_config['关闭按钮']['region'],
                                    self.check_equipment_config['关闭按钮']['points'],
                                    '关闭按钮', 10,
                                    self.get_device(device_id), 5)
        else:
            self.click_time_out(self.check_equipment_config['关闭按钮']['region'],
                                self.check_equipment_config['关闭按钮']['points'],
                                '关闭按钮', 10,
                                self.get_device(device_id), 5)


    # 自动穿戴高级装备
    def check_equipment(self, device_id):
        print('初始化界面')
        self.back_home('否按钮', device_id)
        self.click_time_out(self.check_equipment_config['背包图标']['region'],
                            self.check_equipment_config['背包图标']['points'],
                            '背包图标', 10,
                            self.get_device(device_id), 5)
        self.random_wait(1.5, 2)
        self.random_click('装备区域1', self.check_equipment_config,device_id)
        self.random_wait(1.5, 2)
        self.equipment_proces(device_id)

        self.random_wait(1.5, 2)
        self.click_time_out(self.check_equipment_config['防御按钮']['region'],
                            self.check_equipment_config['防御按钮']['points'],
                            '防御按钮', 10,
                            self.get_device(device_id), 5)

        self.random_wait(1.5, 2)
        self.random_click('装备区域1', self.check_equipment_config,device_id)
        self.random_wait(1.5, 2)
        self.equipment_proces(device_id)

        self.press_esc(1, device_id)
        time.sleep(2)

    def swipe_on_device(self, start_x, start_y, end_x, end_y,
                        duration=0.8, steps=5, device_id=None):
        """
        在指定设备上执行滑动操作
        :param start_x: 起始点X坐标
        :param start_y: 起始点Y坐标
        :param end_x: 结束点X坐标
        :param end_y: 结束点Y坐标
        :param duration: 滑动持续时间(秒)
        :param steps: 滑动步数
        :param device_id: 设备ID（可选）
        """
        # 如果指定了设备ID，则设置当前设备
        if device_id:
            try:
                dev_id = self.get_device(device_id)
                print(dev_id)
                set_current(dev_id)
            except Exception as e:
                print(f"Error setting current device: {e}")

        # 执行滑动操作
        swipe((start_x, start_y), (end_x, end_y), duration=duration, steps=steps)

        # 可选：添加随机等待时间
        self.random_wait(0.5, 1.0)

    # 领取活动奖励 这个是带有滑动功能
    def get_activity_reward2(self, device_id):
        print('初始化界面')
        self.back_home('否按钮', device_id)
        self.random_click('菜单按钮', self.get_activity_reward_config,device_id)
        self.random_wait(1, 1.5)

        self.find_and_click_colors('活动按钮', self.get_activity_reward_config, device_id)
        time.sleep(4)
        # 向上滑动（从底部到顶部）
        self.swipe_on_device(100, 640, 100, 277, duration=0.5, device_id=device_id)
        time.sleep(1)
        self.random_click('全部成长支援', self.get_activity_reward_config,device_id)
        # self.find_and_click_colors('全部成长支援', self.get_activity_reward_config, device_id)
        time.sleep(1)
        self.random_click('领取按钮', self.get_activity_reward_config,device_id)
        time.sleep(1)
        self.press_esc(1, device_id)
        time.sleep(3)
        self.press_esc(1,device_id)

        time.sleep(2)

        # 领取活动奖励 这个是带有滑动功能
    def get_activity_reward(self, device_id):
            print('初始化界面')
            self.back_home('否按钮', device_id)
            self.random_click('菜单按钮', self.get_activity_reward_config, device_id)
            self.random_wait(1, 1.5)

            self.find_and_click_colors('活动按钮', self.get_activity_reward_config, device_id)
            time.sleep(4)

            self.random_click('全部成长支援', self.get_activity_reward_config, device_id)
            # self.find_and_click_colors('全部成长支援', self.get_activity_reward_config, device_id)
            time.sleep(1)
            self.find_and_click_colors('领取按钮', self.get_activity_reward_config, device_id)
            # self.random_click('领取按钮', self.get_activity_reward_config, device_id)
            time.sleep(1)
            self.press_esc(1, device_id)
            time.sleep(3)
            self.press_esc(1, device_id)
            time.sleep(2)

    #购买药品
    def buy_medicine(self, device_id):
        print('初始化界面')
        # self.back_home('否按钮', device_id)
        time.sleep(0.5)
        number = image_to_text([1136, 369, 1181, 387], self.get_device(device_id))
        print(number,'药水数量')
        time.sleep(0.5)
        if int(number)<300:
            print('药水数量大于200')
            self.random_click('菜单按钮', self.star_battle_config, device_id)
            time.sleep(1)
            task_name = ['商店按钮','药水3阶', '选择按钮','购买500','购买','确认购买']
            for name in task_name:
                self.perform_task(name, self.star_battle_config, device_id)
                time.sleep(2)
            self.press_esc(2, device_id)
        else:
            print('药品充足暂不购买')





if __name__ == "__main__":
    controller = GetAfkTimeController()
    # controller.switch_role('角色3', controller.choose_device()[0])
    # print(controller.choose_device())
    # device_id = controller.get_device()
    # controller.check_mail('127.0.0.1:7555')
    # controller.switch_role('角色2','127.0.0.1:7555')
    # controller.elite_dungeons(device_id='192.168.31.14:5005')
    # controller.main_work('192.168.31.239:5004','角色1')
    # print(controller.get_current_grade('角色1'))
    # controller.work2('192.168.31.239:5006','角色1')
    # controller.run_transcript('127.0.0.1:7555','角色1')
    # controller.get_activity_reward('127.0.0.1:7555')
    # controller.adventure_ground('192.168.31.239:5005','角色1')
    # controller.run_star_battle('192.168.31.239:5002','角色1')
    controller.run_star_battle('127.0.0.1:7555', '角色1')
    # controller.everyday_dungeons(device_id='192.168.31.14:5005')
    # controller.buy_medicine('192.168.31.239:5002')
    # controller.get_afk_time(device_id='192.168.31.14:5005')
    # controller.get_activity_reward(device_id='192.168.31.14:5002')
    # print(controller.get_device(controller.choose_device()[0]))
    # controller.get_money('192.168.31.14:5004','角色1')
    # print(result)
    # print(load_role_task_config('config/global_task_config.json'))
