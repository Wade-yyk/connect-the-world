"""Choose one rectangle in each of two photos for A.3."""

import json
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from mark_points import ImagePane


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "correspondences" / "rectification" / "points.json"
ORDER = ("左上", "右上", "右下", "左下")


class RectanglePane(ImagePane):
    def draw(self):
        super().draw()
        if len(self.points) > 1:
            coordinates = [(self.offset_x + (p[0] + 0.5) * self.scale,
                            self.offset_y + (p[1] + 0.5) * self.scale) for p in self.points]
            if len(coordinates) == 4:
                coordinates.append(coordinates[0])
            self.canvas.create_line(*[v for p in coordinates for v in p], fill="#ffe04a", width=2)


class RectificationPicker:
    def __init__(self, root):
        self.root = root
        root.title("A.3 — 自选图片与矩形四角")
        root.geometry("1000x900")
        root.minsize(780, 650)
        self.configuration = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.examples = self.configuration["examples"]
        self.paths = sorted(path for path in (ROOT / "photos").glob("*/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.index = 0
        self.points = []
        toolbar = ttk.Frame(root, padding=10)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="例子：").pack(side="left")
        self.example_choice = ttk.Combobox(toolbar, state="readonly", values=[f"例 {i + 1}" for i in range(len(self.examples))], width=8)
        self.example_choice.current(0)
        self.example_choice.pack(side="left", padx=(0, 16))
        self.example_choice.bind("<<ComboboxSelected>>", self.switch_example)
        ttk.Label(toolbar, text="照片：").pack(side="left")
        self.image_choice = ttk.Combobox(toolbar, state="readonly", values=[p.relative_to(ROOT).as_posix() for p in self.paths], width=25)
        self.image_choice.pack(side="left", padx=(0, 12))
        self.image_choice.bind("<<ComboboxSelected>>", self.switch_image)
        ttk.Button(toolbar, text="撤销 (Ctrl+Z)", command=self.undo).pack(side="left", padx=4)
        ttk.Button(toolbar, text="重新选点", command=self.clear).pack(side="left", padx=4)
        ttk.Button(toolbar, text="恢复全图", command=lambda: self.pane.fit()).pack(side="left", padx=4)
        settings = ttk.Frame(root, padding=(10, 0, 10, 8))
        settings.pack(fill="x")
        self.title = tk.StringVar()
        self.width = tk.StringVar()
        self.height = tk.StringVar()
        for label, variable, width in (("名称：", self.title, 23), ("目标宽：", self.width, 7), ("目标高：", self.height, 7)):
            ttk.Label(settings, text=label).pack(side="left")
            ttk.Entry(settings, textvariable=variable, width=width).pack(side="left", padx=(0, 15))
        ttk.Label(settings, text="尺寸不确定时可以先保持默认。").pack(side="left")
        self.status = ttk.Label(root, anchor="center", font=("Microsoft YaHei UI", 12, "bold"))
        self.status.pack(fill="x", pady=5)
        ttk.Label(root, text="选一个完整矩形的四角：左上 → 右上 → 右下 → 左下。滚轮放大，右键拖动。", anchor="center").pack(fill="x")
        self.pane = RectanglePane(root, "选择四角", self.add)
        self.pane.pack(fill="both", expand=True, padx=12, pady=8)
        controls = ttk.Frame(root, padding=12)
        controls.pack(fill="x")
        ttk.Button(controls, text="保存当前例子的四角", command=self.save).pack(side="left")
        ttk.Button(controls, text="两例保存后：生成 A.3 并更新网页", command=self.generate).pack(side="right")
        self.footer = ttk.Label(root, anchor="center")
        self.footer.pack(fill="x", padx=10, pady=(0, 10))
        root.bind("<Control-z>", lambda _e: self.undo())
        self.load(0)

    def load(self, index):
        self.index = index
        example = self.examples[index]
        self.image_choice.set(example["image"])
        self.title.set(example["title"])
        self.width.set(str(example["target_size"][0]))
        self.height.set(str(example["target_size"][1]))
        self.points = [p.copy() for p in example["source_points"]] if example.get("selected_by") == "user" else []
        self.pane.load(ROOT / example["image"], self.points)
        self.refresh()

    def switch_example(self, _event):
        if self.points != self.saved_points():
            if not messagebox.askyesno("尚未保存", "当前四角尚未保存。仍要切换例子吗？"):
                self.example_choice.current(self.index)
                return
        self.load(self.example_choice.current())

    def saved_points(self):
        example = self.examples[self.index]
        return example["source_points"] if example.get("selected_by") == "user" else []

    def switch_image(self, _event):
        self.points = []
        self.pane.load(ROOT / self.image_choice.get(), self.points)
        self.title.set(f"Rectification {self.index + 1}")
        self.refresh()

    def add(self, point):
        if len(self.points) < 4:
            self.points.append(point)
            self.refresh()

    def undo(self):
        if self.points:
            self.points.pop()
            self.refresh()

    def clear(self):
        self.points.clear()
        self.refresh()

    def refresh(self):
        count = len(self.points)
        instruction = f"请点击第 {count + 1} 个角：{ORDER[count]}" if count < 4 else "四角已选齐，请保存当前例子"
        self.status.configure(text=f"例 {self.index + 1}  |  {count}/4  |  {instruction}")
        self.pane.draw()
        saved = [str(i + 1) for i, item in enumerate(self.examples) if item.get("selected_by") == "user"]
        self.footer.configure(text=f"已保存的自选例子：{', '.join(saved) if saved else '暂无'}。保存后生成 A.3，网页才会使用新结果。")

    def save(self):
        try:
            if len(self.points) != 4:
                raise ValueError("请先按顺序选齐四个角。")
            crosses = []
            for i in range(4):
                a, b, c = (self.points[j % 4] for j in (i, i + 1, i + 2))
                crosses.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
            if min(crosses) <= 1:
                raise ValueError("四角顺序或形状不正确。请按左上、右上、右下、左下选择一个完整矩形。")
            width, height = int(self.width.get()), int(self.height.get())
            if not (80 <= width <= 3000 and 80 <= height <= 3000):
                raise ValueError("目标宽、高请设为 80–3000 像素。")
            example = dict(self.examples[self.index])
            example.update(image=self.image_choice.get(), title=self.title.get().strip() or f"Rectification {self.index + 1}",
                           source_points=[p.copy() for p in self.points], source_size=list(self.pane.image.size),
                           target_size=[width, height], selected_by="user")
            left, top = (width - 80) // 2, (height - 80) // 2
            example["zoom_box"] = [left, top, left + 80, top + 80]
            updated = dict(self.configuration)
            updated["examples"] = self.examples.copy()
            updated["examples"][self.index] = example
            previous = CONFIG.read_text(encoding="utf-8")
            CONFIG.with_name("points.previous.json").write_text(previous, encoding="utf-8")
            temporary = CONFIG.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(CONFIG)
            self.configuration = updated
            self.examples = updated["examples"]
            self.refresh()
            self.status.configure(text=f"例 {self.index + 1} 已保存。切换另一个例子继续选点。")
        except (ValueError, OSError) as error:
            messagebox.showerror("无法保存", str(error))

    def generate(self):
        if any(item.get("selected_by") != "user" for item in self.examples):
            messagebox.showinfo("还需要自选点", "请给两个例子各选四个角，并分别点击保存。")
            return
        if len({item["image"] for item in self.examples}) != len(self.examples):
            messagebox.showerror("照片重复", "A.3 需要至少两张不同照片，请给两个例子选不同照片。")
            return
        if self.points != self.saved_points() or self.image_choice.get() != self.examples[self.index]["image"]:
            messagebox.showinfo("还有未保存修改", "请先保存当前例子，再生成结果。")
            return
        self.status.configure(text="正在生成两种插值结果并更新网页……")
        self.root.update_idletasks()
        result = subprocess.run([sys.executable, str(ROOT / "a3_rectify.py")], cwd=ROOT, capture_output=True, text=True, errors="replace")
        self.refresh()
        if result.returncode:
            messagebox.showerror("生成失败", (result.stderr or result.stdout)[-2500:])
        else:
            self.status.configure(text="A.3 已使用你选的四角更新。刷新网页即可查看。")
            messagebox.showinfo("更新完成", "A.3 已更新为你选择的图片和四角。\n刷新网页的 A.3 部分即可查看。")


if __name__ == "__main__":
    root = tk.Tk()
    ttk.Style().configure(".", font=("Microsoft YaHei UI", 10))
    RectificationPicker(root)
    root.mainloop()
