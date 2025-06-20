import multiprocessing
import subprocess
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait, as_completed
from game_function_module import *
from multiprocessing import cpu_count
from queue import Queue
from threading import Lock, Thread
from PIL import Image
import numpy as np
import cv2
import json
from airtest.core.api import *
import easyocr
import win32gui
import win32con
import win32api


# BATCH_SIZE = 10  # 每批最多处理的设备数量

class GameControlSystem:
    def __init__(self):
        self.row_comboboxes = {}
        self.row_task_config_buttons = {}
        self.root = tk.Tk()
        self.root.title("游戏控制系统")

        # 检查显卡状态
        if not self.check_gpu_status():
            messagebox.showerror("错误", "显卡驱动可能存在问题，请更新显卡驱动或重启电脑")

        self.root.geometry("1500x800")

        # 初始化 ADB 路径
        self.adb_path = None

        # 为每个设备创建独立的控制器
        self.device_controllers = {}
        self.mainline_task_pool = {}

        self.mainline_task_lock = Lock()
        self.mainline_task_started = False

        # 创建顶部菜单
        self.menu_frame = tk.Frame(self.root)
        self.menu_frame.pack(fill=tk.X, padx=5, pady=5)

        # 创建设备菜单
        self.device_button = tk.Menubutton(self.menu_frame, text="设备")
        self.device_button.pack(side=tk.LEFT, padx=2)

        self.device_menu = tk.Menu(self.device_button, tearoff=0)
        self.device_button.configure(menu=self.device_menu)

        self.device_menu.add_command(label="刷新设备", command=self.refresh_devices)
        self.device_menu.add_command(label="开始运行脚本", command=self.start_script)
        self.device_menu.add_command(label="停止运行脚本", command=self.stop_script)
        self.device_menu.add_separator()
        self.device_menu.add_command(label="退出", command=self.root.quit)

        # 创建全局配置按钮
        tk.Button(self.menu_frame, text="全局配置", command=self.open_global_config).pack(side=tk.LEFT, padx=2)

        # 创建【角色配置】按钮
        tk.Button(self.menu_frame, text="角色配置", command=self.open_role_config).pack(side=tk.LEFT, padx=2)

        tk.Label(self.menu_frame, text="并发数量:").pack(side=tk.LEFT, padx=(20, 2))
        self.concurrent_limit_var = tk.StringVar(value="10")  # 默认值设为10
        tk.Entry(self.menu_frame, width=5, textvariable=self.concurrent_limit_var).pack(side=tk.LEFT)

        # 指定 JSON 配置文件的路径
        self.role_task_config_path = "config/role_task_config.json"

        # 角色勾选状态容器（内存中）
        self.role_vars = {}

        # 创建表格
        self.create_table()

        # 创建日志区域
        self.create_log_area()

        # 创建底部配置区域
        self.create_bottom_config()

        # 创建两个不同用途的线程池
        self.io_thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (cpu_count() or 1) * 4),
            thread_name_prefix="io_worker"
        )

        self.cpu_thread_pool = ThreadPoolExecutor(
            max_workers=min(16, (cpu_count() or 1) * 2),
            thread_name_prefix="cpu_worker"
        )
        self.load_adb_path()  # 先尝试加载 ADB 路径
        # 任务队列和状态管理
        self.task_queue = Queue()
        self.device_status = {}
        self.running_tasks = {}

        # 初始化游戏控制器
        self.game_object = None
        # self.initialize_game_controller()

        # 添加设备管理相关的属性
        self.device_locks = {}  # 每个设备的独立锁
        self.connection_retries = {}  # 连接重试次数

    def load_adb_path(self):
        """从配置文件加载 ADB 路径"""
        config_path = "config/adb_path.conf"
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.adb_path = f.read().strip()
                    if not self.adb_path or not os.path.exists(self.adb_path):
                        self.log_text.insert(tk.END, "ADB 路径无效或文件不存在，请选择正确的 ADB 文件\n")
                        self.adb_path = None
                    else:
                        self.log_text.insert(tk.END, f"已加载 ADB 路径: {self.adb_path}\n")
            else:
                self.log_text.insert(tk.END, "未找到 ADB 路径配置文件，请选择 ADB 文件\n")
                self.adb_path = None
        except Exception as e:
            self.log_text.insert(tk.END, f"加载 ADB 路径失败: {e}\n")
            self.adb_path = None

    def initialize_game_controller(self):
        """初始化游戏控制器"""
        try:
            if not self.adb_path or not os.path.exists(self.adb_path):
                messagebox.showwarning("警告", "未找到有效的 ADB 路径，请先选择 ADB 文件")
                self.browse_adb()

            # 获取所有可用设备
            self.device_id = self.get_available_devices()

            # 为每个设备创建独立的控制器
            for device_id in self.device_id:
                try:
                    controller = GetAfkTimeController()
                    controller.device_serial = [device_id]  # 设置单个设备
                    controller.choose_device()  # 初始化设备连接
                    self.device_controllers[device_id] = controller
                except Exception as e:
                    self.log_text.insert(tk.END, f"初始化设备 {device_id} 控制器失败: {str(e)}\n")
                    continue

        except Exception as e:
            messagebox.showerror("错误", f"初始化游戏控制器失败: {str(e)}\n请确保选择了正确的 ADB 文件")
            self.log_text.insert(tk.END, f"初始化游戏控制器失败: {str(e)}\n")
            self.device_controllers = {}
            self.device_id = []

    def get_available_devices(self):
        """获取所有可用的设备列表"""
        try:
            if not self.adb_path or not os.path.exists(self.adb_path):
                raise Exception("ADB 路径无效")
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run([self.adb_path, 'devices'], capture_output=True, text=True, startupinfo=startupinfo,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            lines = result.stdout.strip().split('\n')[1:]
            devices = []

            for line in lines:
                if line.strip():
                    serial, status = line.split()
                    if status == 'device':
                        devices.append(serial)

            return devices
        except Exception as e:
            self.log_text.insert(tk.END, f"获取设备列表失败: {e}\n")
            return []

    def browse_adb(self):
        """打开文件对话框选择ADB可执行文件"""
        file_path = filedialog.askopenfilename(
            title="选择ADB可执行文件",
            filetypes=[("可执行文件", "*.exe"), ("所有文件", "*.*")]
        )
        if file_path:
            self.adb_path = file_path
            self.adb_path_entry.delete(0, tk.END)
            self.adb_path_entry.insert(0, file_path)
            self.save_adb_path()
            # # 重新初始化游戏控制器
            # self.initialize_game_controller()
            # # 刷新设备列表
            # self.refresh_devices()

    def save_adb_path(self):
        """保存当前ADB路径到配置文件"""
        if not self.adb_path:
            messagebox.showerror("错误", "请先选择 ADB 文件")
            return

        # 创建config目录（如果不存在）
        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        config_file = os.path.join(config_dir, "adb_path.conf")
        try:
            with open(config_file, "w") as f:
                f.write(self.adb_path)
            self.log_text.insert(tk.END, f"ADB路径已保存到 {config_file}\n")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存ADB路径失败: {e}")

    def create_table(self):
        """创建表格，第一列作为复选框，并添加垂直滚动条"""
        columns = ('复选框', '状态', '设备ID', '角色1', '角色2', '角色3',
                   '角色4', '角色5', '角色6', '当前角色', '当前角色任务',
                   '当前角色轮换状态', '枫叶币统计', '任务配置')

        # 创建容器 Frame
        table_frame = ttk.Frame(self.root)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        # Treeview 本体
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=85)

        # 垂直滚动条
        yscrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscrollbar.set)
        # 水平滚动条（可选，若列较多可加上）
        xscrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscrollcommand=xscrollbar.set)

        # 使用 grid 布局，让滚动条贴着 Treeview 右侧和下方
        self.tree.grid(row=0, column=0, sticky='nsew')
        yscrollbar.grid(row=0, column=1, sticky='ns')
        xscrollbar.grid(row=1, column=0, sticky='ew')

        # 设置 grid 扩展
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        # 复选框存储
        self.checkbox_vars = {}

        # 单击事件绑定
        self.tree.bind("<ButtonRelease-1>", self.on_checkbox_click)

    def add_device_batches(self, batch_size=20):
        """按批次添加设备到表格中（默认每批20台）"""
        data = load_role_task_config('config/role_task_config.json')
        task = []
        for role_task in data.values():
            temp = []
            if role_task['主线任务']:
                temp.append('主线任务')
            if role_task['日常副本']:
                temp.append('日常副本')
            if role_task['星力战场']:
                temp.append('星力战场')
            task.append(temp)

        end_index = min(self.ui_add_index + batch_size, len(self.initialized_devices))
        for i in range(self.ui_add_index, end_index):
            device_id = self.initialized_devices[i]
            item_id = self.tree.insert('', tk.END,
                                       values=('☐', '已连接', device_id,
                                               task[0] if len(task) > 0 else "",
                                               task[1] if len(task) > 1 else "",
                                               task[2] if len(task) > 2 else "",
                                               task[3] if len(task) > 3 else "",
                                               task[4] if len(task) > 4 else "",
                                               task[5] if len(task) > 5 else "",
                                               '角色1', '', "", '', ''))
            self.checkbox_vars[item_id] = False
            self.root.after(100, self.add_combobox_to_row, item_id)
            self.root.after(100, self.add_task_config_button_to_row, item_id)

        self.ui_add_index = end_index  # ⬅️ 更新当前添加进度

    def add_combobox_to_row(self, item_id):
        """在指定行的'当前角色轮换状态'单元格上叠加一个下拉框"""
        # 列号：'当前角色轮换状态'在 columns 中的第12列，对应 Tkinter 内部标识为 "#12"
        bbox = self.tree.bbox(item_id, "#12")
        if not bbox:
            # 如果此时还未绘制完成，可以再延时尝试一次
            self.root.after(100, self.add_combobox_to_row, item_id)
            return
        x, y, width, height = bbox

        # 创建下拉框，下拉选项为角色1~角色6，默认选中角色1
        combobox = ttk.Combobox(self.tree, values=["角色1", "角色2", "角色3", "角色4", "角色5", "角色6"])
        combobox.place(x=x, y=y, width=width, height=height)
        combobox.set("角色1")

        # 可选：保存下拉框的引用，便于后续获取用户选择的值或更新位置
        if not hasattr(self, 'row_comboboxes'):
            self.row_comboboxes = {}
        self.row_comboboxes[item_id] = combobox

    def update_device_row(self, device_id, current_role, current_role_task, current_role_money=None):
        # 遍历所有行
        for item in self.tree.get_children():
            # 获取这一行所有的值（注意：设备ID在第三列，索引为2）
            values = self.tree.item(item, 'values')
            if values[2] == device_id:
                # 如果你使用了列名标识，可以用 tree.set() 方法：
                self.tree.set(item, column="当前角色", value=current_role)
                self.tree.set(item, column="当前角色任务", value=current_role_task)
                self.tree.set(item, column="枫叶币统计", value=current_role_money)
                # 如果没有使用列名标识，也可以先将整行数据取出，再更新对应索引（假设当前角色索引为9，当前任务索引为10）
                # new_values = list(values)
                # new_values[9] = current_role
                # new_values[10] = current_role_task
                # self.tree.item(item, values=tuple(new_values))
                break  # 找到后就退出循环

    def update_device_row2(self, device_id, role_values):
        """
        根据设备ID更新对应行中"角色1"到"角色6"的信息。

        参数:
          device_id: 要更新的设备ID
          role_values: 包含6个字符串的列表，分别对应角色1～角色6的新值
        """
        # 遍历所有行
        for item in self.tree.get_children():
            # 获取这一行所有的值（注意：设备ID在第三列，索引为2）
            values = self.tree.item(item, 'values')
            if values[2] == device_id:
                # 假设在创建 Treeview 时，你为每一列设置了标识符，例如 "角色1", "角色2", ... "角色6"
                roles = ['角色1', '角色2', '角色3', '角色4', '角色5', '角色6']
                for i, role in enumerate(roles):
                    # 更新每个角色列的值
                    self.tree.set(item, column=role, value=role_values[i])
                break  # 找到对应行后退出循环

    def on_checkbox_click(self, event):
        """监听表格点击事件，切换复选框状态"""
        item_id = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if item_id and column == "#1":
            current_state = self.checkbox_vars[item_id]
            new_state = not current_state
            self.checkbox_vars[item_id] = new_state

            self.tree.item(item_id, values=('☑' if new_state else '☐',) + self.tree.item(item_id, 'values')[1:])

    def create_log_area(self):
        """创建日志区域"""
        tk.Label(self.root, text="日志:").pack(anchor=tk.W, padx=5)
        self.log_text = tk.Text(self.root, height=10)
        self.log_text.pack(fill=tk.X, padx=5, pady=5)

    def create_bottom_config(self):
        """创建底部配置区域"""
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(bottom_frame, text="ADB路径:").pack(side=tk.LEFT)
        self.adb_path_entry = tk.Entry(bottom_frame)
        self.adb_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        # 默认路径
        self.adb_path_entry.insert(0, "D:/投屏/QuickMirror/adb.exe")

        tk.Button(bottom_frame, text="浏览", command=self.browse_adb).pack(side=tk.LEFT, padx=2)
        tk.Button(bottom_frame, text="保存路径", command=self.save_adb_path).pack(side=tk.LEFT, padx=2)

    def refresh_devices(self):
        """刷新设备列表，初始化分页"""
        threading.Thread(target=self._refresh_devices_impl).start()

    def _refresh_devices_impl(self):
        """刷新设备列表并分批添加到 UI"""
        # 清理 UI 组件
        for item_id in self.row_comboboxes:
            if self.row_comboboxes[item_id].winfo_exists():
                self.row_comboboxes[item_id].destroy()
        self.row_comboboxes.clear()

        for item_id in self.row_task_config_buttons:
            if self.row_task_config_buttons[item_id].winfo_exists():
                self.row_task_config_buttons[item_id].destroy()
        self.row_task_config_buttons.clear()

        self.checkbox_vars.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.device_id = self.get_available_devices()

        self.device_controllers.clear()
        self.initialized_devices = []  # 用于记录初始化完成的设备
        self.ui_add_index = 0  # ⬅️ 这个变量记录 UI 添加进度

        def init_device_step(index):
            """初始化一台设备并加入列表"""
            if index >= len(self.device_id):
                return  # 全部完成

            device_id = self.device_id[index]
            try:
                controller = GetAfkTimeController()
                controller.device_serial = [device_id]
                controller.choose_device()
                self.device_controllers[device_id] = controller
                self.initialized_devices.append(device_id)
            except Exception as e:
                self.log_text.insert(tk.END, f"初始化设备 {device_id} 控制器失败: {str(e)}\n")

            if len(self.device_id) <= 20:
                # 小于等于20台，初始化完直接添加一台
                self.add_device_batches(batch_size=1)
            elif len(self.initialized_devices) % 20 == 0 or index == len(self.device_id) - 1:
                # 每20台或最后一台时批量添加
                self.add_device_batches(batch_size=20)

            # 异步继续初始化下一台设备
            self.root.after(10, lambda: init_device_step(index + 1))

        init_device_step(0)

    def get_device(self, device_id):
        """获取指定设备ID的设备对象"""
        try:
            # 确保设备锁存在
            if device_id not in self.device_locks:
                self.device_locks[device_id] = threading.Lock()

            with self.device_locks[device_id]:
                # 检查设备连接状态
                device = self.device_controllers.get(device_id)
                if device:
                    try:
                        # 验证设备连接是否有效
                        device.snapshot()
                        return device
                    except Exception:
                        print(f"设备 {device_id} 连接已失效，尝试重新连接")
                        del self.device_controllers[device_id]

                # 重新连接设备
                retries = self.connection_retries.get(device_id, 0)
                max_retries = 3
                retry_delay = 2  # 重试延迟（秒）

                while retries < max_retries:
                    try:
                        device = connect_device(f"android:///{device_id}")
                        if device:
                            self.device_controllers[device_id] = device
                            self.connection_retries[device_id] = 0
                            print(f"设备 {device_id} 连接成功")
                            return device
                    except Exception as e:
                        print(f"设备 {device_id} 连接失败 (尝试 {retries + 1}/{max_retries}): {e}")
                        retries += 1
                        self.connection_retries[device_id] = retries
                        if retries < max_retries:
                            time.sleep(retry_delay)

                print(f"设备 {device_id} 连接失败，达到最大重试次数")
                return None

        except Exception as e:
            print(f"获取设备 {device_id} 时发生错误: {e}")
            return None

    # 开始脚本
    def start_script(self):
        """优化后的脚本启动函数，实现动态补位并发执行（最大10台同时运行）"""
        running_devices = []
        for item_id in self.tree.get_children():
            if self.checkbox_vars.get(item_id, False):
                values = self.tree.item(item_id, 'values')
                if len(values) > 2:
                    device_id = values[2]
                    running_devices.append(device_id)

        if not running_devices:
            messagebox.showerror('错误', '没有勾选任何设备，无法运行脚本！请勾选需要运行的设备\n')
            self.log_text.insert(tk.END, "没有勾选任何设备，无法运行脚本！\n")
            return

        # 在后台线程中运行，避免阻塞UI
        threading.Thread(target=self._run_dynamic_pool_background, args=(running_devices,)).start()

    def _run_dynamic_pool_background(self, running_devices):
        """动态补位的后台线程函数：保持最多 BATCH_SIZE 个设备同时执行"""
        device_queue = Queue()
        for device in running_devices:
            device_queue.put(device)

        # 线程安全地记录当前正在运行的设备数
        active_count = 0
        active_count_lock = threading.Lock()

        def worker():
            nonlocal active_count
            while not device_queue.empty():
                device = device_queue.get()
                try:
                    with active_count_lock:
                        active_count += 1

                    # 注册设备为运行中
                    self.running_tasks[device] = True

                    self.device_status[device] = {
                        'status': 'pending',
                        'current_role': None,
                        'current_task': None,
                        'start_time': time.time()
                    }
                    self.root.after(0, self.log_text.insert, tk.END, f"设备 {device} 的任务开始执行\n")
                    self.root.after(0, self.update_device_status, device, "运行中")

                    # 执行任务
                    self.device_task_manager(device)

                    self.root.after(0, self.update_device_status, device, "已完成")

                except Exception as e:
                    self.root.after(0, self.log_text.insert, tk.END, f"设备 {device} 执行失败: {e}\n")
                finally:
                    with active_count_lock:
                        active_count -= 1
                    self.running_tasks.pop(device, None)  # 移除任务标记
                    device_queue.task_done()

        # 启动初始 BATCH_SIZE 个线程
        batch_size = int(self.concurrent_limit_var.get())
        for _ in range(min(batch_size, device_queue.qsize())):
            threading.Thread(target=worker, daemon=True).start()

    def device_task_manager(self, device_id):
        """设备任务管理器，实现并发执行"""
        try:
            # 获取设备配置
            config = self.get_device_config(device_id)

            # 启动设备监控
            monitor_thread = threading.Thread(
                target=self.monitor_device_status,
                args=(device_id,),
                daemon=True
            )
            monitor_thread.start()

            # 执行任务序列
            self.perform_tasks(device_id)

        except Exception as e:
            self.log_error(device_id, "任务管理器", str(e))
        finally:
            self.cleanup_device(device_id)

    def task_completed_callback(self, device):
        """任务完成后的回调函数"""
        if device in self.running_tasks:
            del self.running_tasks[device]
            self.log_text.insert(tk.END, f"设备 {device} 的任务已完成！\n")

    def stop_script(self):
        """停止所有勾选的设备的脚本"""
        running_devices = [self.tree.item(item_id, 'values')[2] for item_id, checked in self.checkbox_vars.items() if
                           checked]

        if not running_devices:
            self.log_text.insert(tk.END, "没有设备正在运行，停止无效！\n")
            return

        for device in running_devices:
            if device in self.running_tasks:
                # 取消正在运行的任务
                self.running_tasks[device].cancel()
                self.log_text.insert(tk.END, f"设备 {device} 的脚本已停止！\n")
                del self.running_tasks[device]

    def add_task_config_button_to_row(self, item_id):
        """
        在指定行的"任务配置"单元格上添加按钮，
        该按钮点击时会打开对应设备的任务配置子页面
        """
        # "任务配置"在第14列，对应内部标识为 "#14"
        bbox = self.tree.bbox(item_id, "#14")
        if not bbox:
            # 如果还未绘制完成，延时后重试
            self.root.after(100, self.add_task_config_button_to_row, item_id)
            return
        x, y, width, height = bbox
        # 获取设备ID（假设在第三列，索引为2）
        values = self.tree.item(item_id, 'values')
        device_id = values[2]
        btn = tk.Button(self.tree, text="配置",
                        command=lambda d=device_id: self.open_task_config(d))
        btn.place(x=x, y=y, width=width, height=height)
        self.row_task_config_buttons[item_id] = btn

    def open_task_config(self, device_id):
        """
        打开任务配置子页面，界面与角色配置类似，
        但使用单独的配置文件（以设备ID命名），初始默认什么都未选
        """
        config_window = tk.Toplevel(self.root)
        config_window.title(f"任务配置 - {device_id}")
        config_window.geometry("400x300")
        config_window.grab_set()  # 设置模态

        # 获取设备 ID（去除可能的后缀）
        dev_id = device_id.split(':')[0]
        file_path = f"device_config/task_config_{dev_id}.json"

        # 创建 Notebook，每个页签对应 角色1 ~ 角色6 + 手动配置
        notebook = ttk.Notebook(config_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        # 建立一个字典存储每个角色的 Checkbutton 变量
        role_vars = {
            f"角色{i}": {
                "主线任务": tk.BooleanVar(value=False),
                "日常副本": tk.BooleanVar(value=False),
                "星力战场": tk.BooleanVar(value=False)
            }
            for i in range(1, 7)
        }

        # **额外加入手动配置的 Checkbutton 变量**
        manual_config = {
            "默认配置": tk.BooleanVar(value=True),  # 默认选中
            "手动配置": tk.BooleanVar(value=False)
        }

        # **如果配置文件存在，则读取数据**
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    print(file_path)
                    saved_config = json.load(f)
                # 更新 `role_vars` 默认值
                for role, tasks in saved_config.items():
                    if role in role_vars:  # 确保 JSON 文件中的角色名匹配
                        role_vars[role]["主线任务"].set(tasks.get("主线任务", False))
                        role_vars[role]["日常副本"].set(tasks.get("日常副本", False))
                        role_vars[role]["星力战场"].set(tasks.get("星力战场", False))
                    elif role == "手动配置":  # 读取手动配置的选择
                        manual_config["默认配置"].set(tasks.get("默认配置", True))
                        manual_config["手动配置"].set(tasks.get("手动配置", False))
            except Exception as e:
                self.log_text.insert(tk.END, f"{device_id} 的任务配置加载失败: {e}\n")

        # **创建角色 1-6 任务选项**
        for i in range(1, 7):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=f"角色{i}")

            tk.Checkbutton(tab, text="主线任务", variable=role_vars[f"角色{i}"]["主线任务"]).grid(
                row=0, column=0, padx=10, pady=5, sticky="w")
            tk.Checkbutton(tab, text="日常副本", variable=role_vars[f"角色{i}"]["日常副本"]).grid(
                row=1, column=0, padx=10, pady=5, sticky="w")
            tk.Checkbutton(tab, text="星力战场", variable=role_vars[f"角色{i}"]["星力战场"]).grid(
                row=2, column=0, padx=10, pady=5, sticky="w")

        # **添加"手动配置"选项卡**
        manual_tab = ttk.Frame(notebook)
        notebook.add(manual_tab, text="手动配置")

        # **手动配置的复选框**
        default_check = tk.Checkbutton(manual_tab, text="默认配置", variable=manual_config["默认配置"])
        default_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        manual_check = tk.Checkbutton(manual_tab, text="手动配置", variable=manual_config["手动配置"])
        manual_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        # **保证只能选一个**
        def toggle_manual_config(*args):
            if manual_config["默认配置"].get():
                manual_config["手动配置"].set(False)
            elif manual_config["手动配置"].get():
                manual_config["默认配置"].set(False)

        manual_config["默认配置"].trace_add("write", toggle_manual_config)
        manual_config["手动配置"].trace_add("write", toggle_manual_config)

        # **保存任务配置**
        def save_task_config():
            config_data = {
                role: {
                    "主线任务": vars_dict["主线任务"].get(),
                    "日常副本": vars_dict["日常副本"].get(),
                    "星力战场": vars_dict["星力战场"].get()
                }
                for role, vars_dict in role_vars.items()
            }

            # 保存手动配置的选择
            config_data["手动配置"] = {
                "默认配置": manual_config["默认配置"].get(),
                "手动配置": manual_config["手动配置"].get()
            }

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                if config_data['手动配置']['手动配置']:
                    all_roles = ["角色1", "角色2", "角色3", "角色4", "角色5", "角色6"]
                    role_config = []
                    custom_config = load_role_task_config(file_path)  # 如果是手动配置就加载单独设备的config文件
                    for role in all_roles:
                        role_config.append({k: v for k, v in custom_config.get(role, {}).items() if k != "手动配置"})
                    task_2 = []
                    for role_task in role_config:
                        temp = []
                        if role_task['主线任务']:
                            temp.append('主线任务')
                        if role_task['日常副本']:
                            temp.append('日常副本')
                        if role_task['星力战场']:
                            temp.append('星力战场')
                        task_2.append(temp)
                    self.update_device_row2(device_id, task_2)
                else:
                    data = load_role_task_config('config/role_task_config.json')
                    task = []
                    for role_task in data.values():
                        temp = []
                        if role_task['主线任务']:
                            temp.append('主线任务')
                        if role_task['日常副本']:
                            temp.append('日常副本')
                        if role_task['星力战场']:
                            temp.append('星力战场')
                        task.append(temp)
                    self.update_device_row2(device_id, task)
                self.log_text.insert(tk.END, f"{device_id} 的任务配置已保存。\n")


            except Exception as e:
                self.log_text.insert(tk.END, f"{device_id} 的任务配置保存失败: {e}\n")

            config_window.destroy()

        tk.Button(config_window, text="保存", command=save_task_config).pack(pady=10)

    def open_global_config(self):
        """创建全局任务配置窗口，只包含一个「全局任务配置」标签页，并从配置文件中加载默认值"""

        # 尝试从配置文件读取默认配置
        config_data = {}
        try:
            with open("config/global_task_config.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            self.log_text.insert(tk.END, f"读取全局任务配置失败: {e}\n")

        # 定义各任务默认配置（如果配置文件中不存在则使用预设的默认值）
        default_fatigue = config_data.get("自动领取疲劳时间", {"enabled": True, "interval": "10"})
        default_daily = config_data.get("自动领取日常物品", {"enabled": True, "interval": "10"})
        default_bag = config_data.get("自动整理背包装备", {"enabled": True, "interval": "10"})
        default_equip = config_data.get("自动整备高级装备", {"enabled": True, "interval": "10"})
        default_upgrade = config_data.get("自动装备检测升级", {"enabled": True, "interval": "10"})
        default_activity = config_data.get("自动检查活动奖励", {"enabled": True, "interval": "10"})

        # 创建配置窗口
        config_window = tk.Toplevel(self.root)
        config_window.title("全局任务配置")
        config_window.geometry("500x300")
        config_window.grab_set()  # 设置模态

        # 创建 Notebook（选项卡），只添加一个「全局任务配置」Tab
        notebook = ttk.Notebook(config_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        global_task_tab = ttk.Frame(notebook)
        notebook.add(global_task_tab, text="全局任务配置")

        # 用来存储各个复选框的状态、执行间隔等变量
        # 通过配置文件中的数据设置默认值
        self.check_auto_fatigue = tk.BooleanVar(value=default_fatigue.get("enabled", True))
        self.check_auto_daily = tk.BooleanVar(value=default_daily.get("enabled", True))
        self.check_auto_bag = tk.BooleanVar(value=default_bag.get("enabled", True))
        self.check_auto_equip = tk.BooleanVar(value=default_equip.get("enabled", True))
        self.check_auto_upgrade = tk.BooleanVar(value=default_upgrade.get("enabled", True))
        self.check_auto_activity = tk.BooleanVar(value=default_activity.get("enabled", True))

        # 创建输入框，并填入默认的间隔值
        self.entry_auto_fatigue = tk.Entry(global_task_tab, width=10)
        self.entry_auto_fatigue.insert(0, default_fatigue.get("interval", "10"))

        self.entry_auto_daily = tk.Entry(global_task_tab, width=10)
        self.entry_auto_daily.insert(0, default_daily.get("interval", "10"))

        self.entry_auto_bag = tk.Entry(global_task_tab, width=10)
        self.entry_auto_bag.insert(0, default_bag.get("interval", "10"))

        self.entry_auto_equip = tk.Entry(global_task_tab, width=10)
        self.entry_auto_equip.insert(0, default_equip.get("interval", "10"))

        self.entry_auto_upgrade = tk.Entry(global_task_tab, width=10)
        self.entry_auto_upgrade.insert(0, default_upgrade.get("interval", "10"))

        self.entry_auto_activity = tk.Entry(global_task_tab, width=10)
        self.entry_auto_activity.insert(0, default_activity.get("interval", "10"))

        # 布局：每行一个 Checkbutton + Label + Entry + "秒执行一次"
        # 1) 自动领取疲劳时间
        row = 0
        tk.Checkbutton(global_task_tab, variable=self.check_auto_fatigue) \
            .grid(row=row, column=0, padx=5, pady=5, sticky="w")
        tk.Label(global_task_tab, text="自动领取挂机时间") \
            .grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.entry_auto_fatigue.grid(row=row, column=2, padx=5, pady=5)
        tk.Label(global_task_tab, text="秒执行一次") \
            .grid(row=row, column=3, padx=5, pady=5, sticky="w")

        # 2) 自动领取日常物品
        row += 1
        tk.Checkbutton(global_task_tab, variable=self.check_auto_daily) \
            .grid(row=row, column=0, padx=5, pady=5, sticky="w")
        tk.Label(global_task_tab, text="自动领取日常物品") \
            .grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.entry_auto_daily.grid(row=row, column=2, padx=5, pady=5)
        tk.Label(global_task_tab, text="秒执行一次") \
            .grid(row=row, column=3, padx=5, pady=5, sticky="w")

        # 3) 自动整理背包装备
        row += 1
        tk.Checkbutton(global_task_tab, variable=self.check_auto_bag) \
            .grid(row=row, column=0, padx=5, pady=5, sticky="w")
        tk.Label(global_task_tab, text="自动整理背包装备") \
            .grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.entry_auto_bag.grid(row=row, column=2, padx=5, pady=5)
        tk.Label(global_task_tab, text="秒执行一次") \
            .grid(row=row, column=3, padx=5, pady=5, sticky="w")

        # 4) 自动整备高级装备
        row += 1
        tk.Checkbutton(global_task_tab, variable=self.check_auto_equip) \
            .grid(row=row, column=0, padx=5, pady=5, sticky="w")
        tk.Label(global_task_tab, text="自动整备高级装备") \
            .grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.entry_auto_equip.grid(row=row, column=2, padx=5, pady=5)
        tk.Label(global_task_tab, text="秒执行一次") \
            .grid(row=row, column=3, padx=5, pady=5, sticky="w")

        # 5) 自动装备检测升级
        row += 1
        tk.Checkbutton(global_task_tab, variable=self.check_auto_upgrade) \
            .grid(row=row, column=0, padx=5, pady=5, sticky="w")
        tk.Label(global_task_tab, text="自动装备检测升级") \
            .grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.entry_auto_upgrade.grid(row=row, column=2, padx=5, pady=5)
        tk.Label(global_task_tab, text="秒执行一次") \
            .grid(row=row, column=3, padx=5, pady=5, sticky="w")

        # 6) 自动检查活动奖励
        row += 1
        tk.Checkbutton(global_task_tab, variable=self.check_auto_activity) \
            .grid(row=row, column=0, padx=5, pady=5, sticky="w")
        tk.Label(global_task_tab, text="自动检查活动奖励") \
            .grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.entry_auto_activity.grid(row=row, column=2, padx=5, pady=5)
        tk.Label(global_task_tab, text="秒执行一次") \
            .grid(row=row, column=3, padx=5, pady=5, sticky="w")

        # 底部保存按钮
        tk.Button(config_window, text="保存", command=self.save_global_config) \
            .pack(pady=10)

    def save_global_config(self):
        """点击保存时，获取所有复选框与间隔的值，并保存为 JSON 文件到 template/ 目录下"""
        # 获取每个复选框的状态
        fatigue_enabled = self.check_auto_fatigue.get()
        daily_enabled = self.check_auto_daily.get()
        bag_enabled = self.check_auto_bag.get()
        equip_enabled = self.check_auto_equip.get()
        upgrade_enabled = self.check_auto_upgrade.get()
        activity_enabled = self.check_auto_activity.get()

        # 获取每个间隔的值（字符串）
        fatigue_interval = self.entry_auto_fatigue.get()
        daily_interval = self.entry_auto_daily.get()
        bag_interval = self.entry_auto_bag.get()
        equip_interval = self.entry_auto_equip.get()
        upgrade_interval = self.entry_auto_upgrade.get()
        activity_interval = self.entry_auto_activity.get()

        # 构造一个字典保存所有配置
        config_data = {
            "自动领取疲劳时间": {
                "enabled": fatigue_enabled,
                "interval": fatigue_interval
            },
            "自动领取日常物品": {
                "enabled": daily_enabled,
                "interval": daily_interval
            },
            "自动整理背包装备": {
                "enabled": bag_enabled,
                "interval": bag_interval
            },
            "自动整备高级装备": {
                "enabled": equip_enabled,
                "interval": equip_interval
            },
            "自动装备检测升级": {
                "enabled": upgrade_enabled,
                "interval": upgrade_interval
            },
            "自动检查活动奖励": {
                "enabled": activity_enabled,
                "interval": activity_interval
            }
        }

        # 保存配置到 JSON 文件
        try:
            with open("config/global_task_config.json", "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            # 同时在日志框中显示成功信息
            self.log_text.insert(tk.END, "全局任务配置已保存到 template/global_task_config.json\n")
        except Exception as e:
            self.log_text.insert(tk.END, f"保存全局任务配置失败: {e}\n")

    def open_role_config(self):
        """
        打开角色配置窗口
        - 仅包含 角色1~角色6 这几个标签页
        - 每个标签页中有三个 Checkbutton: 主线任务、日常副本、星力战场
        - 在"主线任务"复选框下方添加一个等级输入框
        - 底部有一个"保存"按钮
        - 从 template/role_task_config.json 读取默认勾选状态，并从 template/等级配置.json 读取等级数据
        """
        config_window = tk.Toplevel(self.root)
        config_window.title("角色配置")
        # 增加窗口高度以容纳等级输入框
        config_window.geometry("400x350")
        config_window.grab_set()  # 模态

        # 先加载角色配置 JSON 文件中的默认配置
        default_config = load_role_task_config(self.role_task_config_path)

        # 尝试加载等级配置数据，若文件不存在或解析失败，则设为空字典
        try:
            with open("config/等级配置.json", "r", encoding="utf-8") as f_level:
                level_config = json.load(f_level)
        except Exception:
            level_config = {}

        # 创建 Notebook（只添加角色1 ~ 角色6）
        notebook = ttk.Notebook(config_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        # 初始化 self.role_vars 并将默认值赋给每个角色，同时初始化 self.level_entries 存放等级输入框对象
        self.role_vars.clear()
        self.level_entries = {}
        # default_config 形如:
        # {
        #   "角色1": {"主线任务": True, "日常副本": True, "星力战场": True},
        #   ...
        # }
        for i in range(1, 7):
            role_key = f"角色{i}"
            role_data = default_config.get(role_key, {"主线任务": False, "日常副本": False, "星力战场": False})
            self.role_vars[i] = {
                "main": tk.BooleanVar(value=role_data.get("主线任务", False)),
                "daily": tk.BooleanVar(value=role_data.get("日常副本", False)),
                "starforce": tk.BooleanVar(value=role_data.get("星力战场", False))
            }

        # 为角色1~角色6创建标签页
        for i in range(1, 7):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=f"角色{i}")

            row_idx = 0
            # 主线任务 Checkbutton
            tk.Checkbutton(tab, text="主线任务", variable=self.role_vars[i]["main"]).grid(
                row=row_idx, column=0, padx=10, pady=5, sticky="w"
            )
            row_idx += 1

            # 在主线任务复选框下方添加等级输入框
            tk.Label(tab, text="等级:").grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
            level_entry = tk.Entry(tab)
            level_entry.grid(row=row_idx, column=1, padx=10, pady=5, sticky="w")
            # 如果等级配置中存在该角色的数据，则写入输入框；否则默认为空或 "0"
            current_level = level_config.get(f"角色{i}", {}).get("等级", "0")
            level_entry.insert(0, current_level)
            self.level_entries[i] = level_entry
            row_idx += 1

            # 日常副本 Checkbutton
            tk.Checkbutton(tab, text="日常副本", variable=self.role_vars[i]["daily"]).grid(
                row=row_idx, column=0, padx=10, pady=5, sticky="w"
            )
            row_idx += 1

            # 星力战场 Checkbutton
            tk.Checkbutton(tab, text="星力战场", variable=self.role_vars[i]["starforce"]).grid(
                row=row_idx, column=0, padx=10, pady=5, sticky="w"
            )

        # 底部保存按钮
        tk.Button(config_window, text="保存", command=self.save_role_config).pack(pady=10)

    def save_role_config(self):
        """
        将当前界面的角色勾选状态写回 template/role_task_config.json 文件，
        同时将每个角色等级输入框中的数据保存到 template/等级配置.json 文件，
        并在保存后更新等级输入框显示最新数据
        """
        # 整理角色勾选状态数据
        config_data = {}
        for i in range(1, 7):
            role_key = f"角色{i}"
            main_checked = self.role_vars[i]["main"].get()
            daily_checked = self.role_vars[i]["daily"].get()
            starforce_checked = self.role_vars[i]["starforce"].get()

            config_data[role_key] = {
                "主线任务": main_checked,
                "日常副本": daily_checked,
                "星力战场": starforce_checked
            }

            # 同时打印到日志区
            self.log_text.insert(
                tk.END,
                f"{role_key} 配置: 主线任务={main_checked}, 日常副本={daily_checked}, 星力战场={starforce_checked}\n"
            )

        try:
            # 保存角色任务配置到 JSON 文件
            with open(self.role_task_config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            self.log_text.insert(tk.END, "角色配置已保存至 JSON 文件！\n")

            # -------------------------------
            # 保存等级配置数据
            level_config = {}
            for i in range(1, 7):
                level = self.level_entries[i].get().strip()
                # 如果输入为空，默认为 "0"
                level_config[f"角色{i}"] = {"等级": level if level else "0"}
            with open("config/等级配置.json", "w", encoding="utf-8") as f_level:
                json.dump(level_config, f_level, ensure_ascii=False, indent=2)
            self.log_text.insert(tk.END, "等级配置已保存至 JSON 文件！\n")
            # -------------------------------

            # 更新每个等级输入框显示保存后的最新数据
            for i in range(1, 7):
                current_level = level_config[f"角色{i}"]["等级"]
                self.level_entries[i].delete(0, tk.END)
                self.level_entries[i].insert(0, current_level)

            # 根据最新保存的角色配置更新设备信息
            for id in self.device_id:
                data = load_role_task_config('config/role_task_config.json')
                task = []
                for role_task in data.values():
                    temp = []
                    if role_task['主线任务']:
                        temp.append('主线任务')
                    if role_task['日常副本']:
                        temp.append('日常副本')
                    if role_task['星力战场']:
                        temp.append('星力战场')
                    task.append(temp)
                self.update_device_row2(id, task)
        except Exception as e:
            self.log_text.insert(tk.END, f"保存角色配置失败: {e}\n")

    def run_global_tasks_sequentially(self, device_id, controller1):
        config = load_role_task_config('config/global_task_config.json')
        task_mapping = [
            ("自动领取疲劳时间", controller1.get_afk_time),
            ("自动领取日常物品", controller1.check_mail),
            ("自动整理背包装备", controller1.clear_bug),
            ("自动装备检测升级", controller1.skil_check),
            ("自动整备高级装备", controller1.check_equipment),
            ("自动检查活动奖励", controller1.get_activity_reward),
        ]

        """根据配置执行任务，每个任务都传递device_id参数"""
        for task_name, task_func in task_mapping:
            task_config = config.get(task_name)

            if not task_config:
                print(f"设备{device_id}: 警告 - 任务 '{task_name}' 在配置中未找到，跳过执行")
                continue

            if task_config.get('enabled', False):

                print(f"设备{device_id}: 开始执行任务: {task_name}")
                try:
                    # 传递参数：device_id
                    task_func(device_id)
                except Exception as e:
                    print(f"设备{device_id}: 任务 '{task_name}' 执行失败: {str(e)}")
            else:
                print(f"设备{device_id}: 任务 '{task_name}' 已禁用，跳过执行")

    def perform_tasks(self, device_id):
        """执行设备任务，实现按顺序执行任务"""
        try:
            # 获取当前设备的控制器
            controller = self.device_controllers.get(device_id)
            if not controller:
                self.log_error(device_id, "执行任务", "未找到设备控制器")
                return

            # 更新设备状态为运行中
            self.update_device_status(device_id, "运行中")

            role_list = load_role_task_config('config/role_task_config.json')
            dev_id = device_id.split(':')[0]
            file_path = f"device_config/task_config_{dev_id}.json"

            # 读取设备特定的配置
            custom_config = {}
            use_custom_config = False

            if os.path.exists(file_path):
                custom_config = load_role_task_config(file_path)
                use_custom_config = custom_config.get('手动配置', {}).get("手动配置", False)

            # 默认角色顺序
            all_roles = ["角色1", "角色2", "角色3", "角色4", "角色5", "角色6"]

            # 获取当前设备行对应的 item_id
            target_item = None
            for item in self.tree.get_children():
                values = self.tree.item(item, 'values')
                if values[2] == device_id:
                    target_item = item
                    break

            # 获取用户选择的角色
            start_role = "角色1"
            if target_item and hasattr(self, 'row_comboboxes'):
                combobox = self.row_comboboxes.get(target_item)
                if combobox:
                    start_role = combobox.get()

            # print(start_role)
            # 重新排序角色列表
            if start_role in all_roles:
                start_index = all_roles.index(start_role)
                reordered_roles = all_roles[start_index:] + all_roles[:start_index]
            else:
                reordered_roles = all_roles

            # 重新构造任务列表
            reordered_tasks = []
            for role in reordered_roles:
                if use_custom_config:
                    role_config = {k: v for k, v in custom_config.get(role, {}).items() if k != "手动配置"}
                else:
                    role_config = role_list.get(role, {"主线任务": False, "日常副本": False, "星力战场": False})
                reordered_tasks.append((role, role_config))
            # print(reordered_tasks)
            # 遍历角色，依次执行任务
            for role_key, role_value in reordered_tasks:
                if self.should_stop(device_id):
                    break

                try:
                    # 使用当前设备的控制器切换角色
                    device = self.get_device(device_id)
                    if not device:
                        print(f"未找到设备: {device_id}")
                        continue

                    # 更新当前角色信息
                    self.update_device_status(device_id, f"切换到{role_key}")
                    try:
                        controller.switch_role(role_key, device_id)
                    except Exception as e:
                        self.log_text.insert(tk.END, f"设备{device_id}: 切换角色失败: {str(e)}\n")
                        continue
                    current_money = count_money(device_id)
                    self.update_device_row(device_id, role_key, None, current_money)

                    # 创建任务列表
                    tasks = []

                    self.run_global_tasks_sequentially(device_id, controller)

                    # 判断并添加任务（主线任务 → 日常副本 → 星力战场）
                    if role_value.get('主线任务'):
                        tasks.append(('主线任务', controller.main_work))
                    if role_value.get('日常副本'):
                        tasks.append(('日常副本', controller.run_transcript))
                    if role_value.get('星力战场'):
                        tasks.append(('星力战场', controller.run_star_battle))

                    # 顺序执行任务
                    for task_name, task_func in tasks:
                        try:
                            self.update_device_row(device_id, role_key, f"开始执行{task_name}")
                            # self.update_device_status(device_id, f"{role_key}开始执行{task_name}")
                            # task_func(device_id, role_key,self.log_text)  # **同步调用，按顺序执行**
                            task_func(device_id, role_key)
                            # self.update_device_status(device_id, f"{role_key}的{task_name}完成")
                            self.update_device_row(device_id, role_key, f"{task_name}完成")
                        except Exception as e:
                            self.log_error(device_id, f"{role_key}的{task_name}", str(e))
                            # 任务失败时继续执行下一个，不中断整个流程

                except Exception as e:
                    self.log_error(device_id, role_key, str(e))
                    continue

            # 更新设备状态为已完成
            self.update_device_status(device_id, "已完成")

        except Exception as e:
            self.log_error(device_id, "执行任务", str(e))
            # 发生异常时更新状态为错误
            self.update_device_status(device_id, f"错误: {str(e)}")

    def run(self):
        self.root.mainloop()

    def __del__(self):
        """确保程序退出时关闭线程池"""
        if hasattr(self, 'io_thread_pool'):
            self.io_thread_pool.shutdown(wait=True)
        if hasattr(self, 'cpu_thread_pool'):
            self.cpu_thread_pool.shutdown(wait=True)

    def monitor_device_status(self, device_id):
        """监控设备状态"""
        consecutive_failures = 0
        max_consecutive_failures = 5  # 允许的最大连续失败次数
        check_interval = 5  # 基础检查间隔（秒）

        while device_id in self.running_tasks:
            try:
                # 检查设备连接状态
                if not self.check_device_connection(device_id):
                    consecutive_failures += 1
                    self.log_text.insert(tk.END,
                                         f"设备 {device_id} 连接检查失败 ({consecutive_failures}/{max_consecutive_failures})\n")

                    if consecutive_failures >= max_consecutive_failures:
                        # 尝试重新连接
                        if not self.handle_device_disconnection(device_id):
                            # 如果重连失败，增加检查间隔
                            check_interval = min(check_interval * 2, 60)  # 最大间隔60秒
                        else:
                            # 重连成功，重置计数器和间隔
                            consecutive_failures = 0
                            check_interval = 5
                else:
                    # 连接正常，重置计数器和间隔
                    consecutive_failures = 0
                    check_interval = 5

                    # 更新设备状态为运行中
                    if self.device_status.get(device_id, {}).get('status') == "等待重连":
                        self.update_device_status(device_id, "运行中")

                # 检查资源使用情况
                if self.check_resource_usage() > 90:  # CPU或内存使用率超过90%
                    self.handle_high_resource_usage(device_id)

                # 更新UI状态
                self.update_ui_status(device_id)

                # 动态调整的等待时间
                time.sleep(check_interval)

            except Exception as e:
                self.log_error(device_id, "监控", str(e))
                consecutive_failures += 1
                time.sleep(check_interval)

    def check_resource_usage(self):
        """检查系统资源使用情况"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            return max(cpu_percent, memory_percent)
        except:
            return 0

    def handle_high_resource_usage(self, device_id):
        """处理高资源使用情况"""
        messagebox.showerror(f"警告：系统资源使用率过高，设备 {device_id} 任务执行可能受影响\n")
        self.log_text.insert(tk.END, f"警告：系统资源使用率过高，设备 {device_id} 任务执行可能受影响\n")
        # 可以考虑暂停一些非关键任务

    def handle_device_disconnection(self, device_id):
        """处理设备断开连接"""
        self.log_text.insert(tk.END, f"设备 {device_id} 连接断开，尝试重新连接...\n")

        max_retries = 10  # 增加重试次数
        retry_count = 0
        initial_delay = 5  # 初始等待时间（秒）
        max_delay = 30  # 最大等待时间（秒）

        while retry_count < max_retries:
            try:
                # 尝试重新连接设备
                if self.reconnect_device(device_id):
                    # 验证连接是否真的成功
                    if self.check_device_connection(device_id):
                        self.log_text.insert(tk.END, f"设备 {device_id} 重新连接成功\n")
                        # 重新初始化设备控制器
                        try:
                            controller = GetAfkTimeController()
                            controller.device_serial = [device_id]
                            controller.choose_device()
                            self.device_controllers[device_id] = controller
                            self.update_device_status(device_id, "运行中")
                            return True
                        except Exception as e:
                            self.log_text.insert(tk.END, f"设备 {device_id} 控制器初始化失败: {e}\n")

                # 使用指数退避策略增加等待时间
                wait_time = min(initial_delay * (2 ** retry_count), max_delay)
                self.log_text.insert(tk.END, f"第 {retry_count + 1} 次重连失败，等待 {wait_time} 秒后重试...\n")
                time.sleep(wait_time)

            except Exception as e:
                self.log_error(device_id, "重连", str(e))

            retry_count += 1

        # 达到最大重试次数后，更新状态但不停止任务
        self.log_text.insert(tk.END, f"设备 {device_id} 重连失败，继续监控...\n")
        self.update_device_status(device_id, "等待重连")
        return False

    def update_ui_status(self, device_id):
        """更新UI状态（使用after方法避免阻塞）"""

        def _update():
            try:
                status = self.device_status.get(device_id, {})
                self.tree.set(device_id, "状态", status.get('status', '未知'))
                self.tree.set(device_id, "当前角色", status.get('current_role', ''))
                self.tree.set(device_id, "当前角色任务", status.get('current_task', ''))  # 更新当前角色任务current_task
            except Exception as e:
                print(f"UI更新错误: {e}")

        self.root.after(0, _update, )

    def update_device_status(self, device_id, status):
        """
        更新设备状态，确保状态更新是线程安全的
        """
        try:
            # 更新内部状态字典
            if device_id not in self.device_status:
                self.device_status[device_id] = {}

            self.device_status[device_id].update({
                'status': status,
                'update_time': time.time()
            })

            # 使用线程安全的方式更新UI
            def _update():
                try:
                    # 遍历查找对应设备的行
                    for item in self.tree.get_children():
                        values = self.tree.item(item, 'values')
                        if values[2] == device_id:  # 设备ID在第三列
                            self.tree.set(item, "状态", status)
                            break

                    # 更新日志
                    self.log_text.insert(tk.END, f"设备 {device_id} 状态更新为: {status}\n")
                    self.log_text.see(tk.END)  # 滚动到最新日志

                except Exception as e:
                    print(f"UI更新错误: {e}")

            # 使用after方法在主线程中更新UI
            self.root.after(0, _update)

        except Exception as e:
            print(f"更新设备状态失败: {e}")

    def get_device_config(self, device_id):
        """
        获取设备配置

        参数:
            device_id: 设备ID
        返回:
            设备配置字典
        """
        try:
            dev_id = device_id.split(':')[0]
            file_path = f"device_config/task_config_{dev_id}.json"

            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                # 如果没有特定配置，返回默认配置
                return load_role_task_config('config/role_task_config.json')
        except Exception as e:
            self.log_text.insert(tk.END, f"读取设备 {device_id} 配置失败: {e}\n")
            return {}

    def should_stop(self, device_id):
        """
        检查设备是否应该停止任务
        修改为只检查当前设备的状态，不影响其他设备
        """
        return device_id not in self.running_tasks or self.device_status.get(device_id, {}).get('status') == "已停止"

    def log_error(self, device_id, context, error_msg):
        """
        记录错误信息

        参数:
            device_id: 设备ID
            context: 错误发生的上下文
            error_msg: 错误信息
        """
        error_text = f"设备 {device_id} 在{context}时发生错误: {error_msg}\n"
        self.root.after(0, lambda: self.log_text.insert(tk.END, error_text))

    def cleanup_device(self, device_id):
        """
        清理设备相关资源
        修改为只清理当前设备的资源，不影响其他设备
        """
        try:
            if device_id in self.running_tasks:
                del self.running_tasks[device_id]
            if device_id in self.device_status:
                del self.device_status[device_id]
            self.update_device_status(device_id, "已停止")
        except Exception as e:
            print(f"清理设备资源失败: {e}")

    def check_device_connection(self, device_id):
        """
        检查设备连接状态

        参数:
            device_id: 设备ID
        返回:
            布尔值，True表示连接正常，否则False
        """
        try:
            if not self.adb_path or not os.path.exists(self.adb_path):
                self.log_text.insert(tk.END, f"ADB路径无效或不存在: {self.adb_path}\n")
                return False

            # 使用完整的 adb 路径执行命令
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run(
                [self.adb_path, "-s", device_id, "get-state"],
                capture_output=True,
                text=True,
                timeout=5,
                startupinfo=startupinfo,  # 新增
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            # 检查命令执行结果
            if result.returncode == 0:
                state = result.stdout.strip()
                if state == "device":
                    return True
                else:
                    self.log_text.insert(tk.END, f"设备 {device_id} 状态异常: {state}\n")
                    return False
            else:
                self.log_text.insert(tk.END, f"检查设备 {device_id} 状态失败: {result.stderr}\n")
                return False

        except subprocess.TimeoutExpired:
            self.log_text.insert(tk.END, f"检查设备 {device_id} 状态超时\n")
            return False
        except Exception as e:
            self.log_text.insert(tk.END, f"检查设备 {device_id} 状态时发生错误: {str(e)}\n")
            return False

    def reconnect_device(self, device_id):
        """重新连接设备"""
        try:
            if not self.adb_path or not os.path.exists(self.adb_path):
                self.log_text.insert(tk.END, f"ADB路径无效或不存在: {self.adb_path}\n")
                return False

            # 创建 startupinfo 对象
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            # 判断是否为网络设备
            if ':' in device_id:
                # 先断开该设备连接
                subprocess.run(
                    [self.adb_path, "disconnect", device_id],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    startupinfo=startupinfo,  # 添加 startupinfo
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                time.sleep(2)  # 等待断开连接

                # 尝试重新连接
                result = subprocess.run(
                    [self.adb_path, "connect", device_id],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    startupinfo=startupinfo,  # 添加 startupinfo
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

                # 检查连接结果
                if "connected" in result.stdout.lower():
                    self.log_text.insert(tk.END, f"设备 {device_id} 重新连接成功\n")
                    return True
                else:
                    self.log_text.insert(tk.END, f"设备 {device_id} 重新连接失败: {result.stdout}\n")
                    return False
            else:
                # 对于USB设备，调用adb usb命令进行重置
                result = subprocess.run(
                    [self.adb_path, "-s", device_id, "usb"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    startupinfo=startupinfo,  # 添加 startupinfo
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

                if result.returncode == 0:
                    self.log_text.insert(tk.END, f"USB设备 {device_id} 重置成功\n")
                    return True
                else:
                    self.log_text.insert(tk.END, f"USB设备 {device_id} 重置失败: {result.stderr}\n")
                    return False

        except subprocess.TimeoutExpired:
            self.log_text.insert(tk.END, f"重新连接设备 {device_id} 超时\n")
            return False
        except Exception as e:
            self.log_text.insert(tk.END, f"重新连接设备 {device_id} 时发生错误: {str(e)}\n")
            return False

    def stop_device_tasks(self, device_id):
        """
        停止设备的所有任务

        参数:
            device_id: 设备ID
        """
        try:
            if device_id in self.running_tasks:
                self.running_tasks[device_id].cancel()
                del self.running_tasks[device_id]
            self.update_device_status(device_id, "已停止")
        except Exception as e:
            self.log_error(device_id, "停止任务", str(e))

    def check_gpu_status(self):
        """检查显卡状态"""
        try:
            import win32gui
            import win32con
            import win32api

            # 检查显卡驱动状态
            hdc = win32gui.GetDC(0)
            if hdc:
                win32gui.ReleaseDC(0, hdc)
                return True
        except Exception as e:
            print(f"显卡检查失败: {e}")
            return False


if __name__ == "__main__":
    app = GameControlSystem()
    app.run()

# 目前最好，这个是分批次运行版本
