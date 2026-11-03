"""Manual point correspondences for Assignment 3. Requires Pillow and Tkinter."""

import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageOps, ImageTk


ROOT = Path(__file__).resolve().parent


class ImagePane(ttk.Frame):
    def __init__(self, parent, title, on_point):
        super().__init__(parent)
        self.on_point = on_point
        self.image = None
        self.points = []
        self.scale = 1.0
        self.offset_x = self.offset_y = 0.0
        self.drag = None
        self.heading = ttk.Label(self, text=title, anchor="center")
        self.heading.pack(fill="x", pady=5)
        self.canvas = tk.Canvas(self, background="#20252d", highlightthickness=2)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.resize)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-3>", self.pan_start)
        self.canvas.bind("<B3-Motion>", self.pan_move)
        self.canvas.bind("<Motion>", self.motion)
        self.coordinate_label = ttk.Label(self, text="x: —   y: —", anchor="center")
        self.coordinate_label.pack(fill="x", pady=4)

    def load(self, path, points):
        with Image.open(path) as source:
            self.image = ImageOps.exif_transpose(source).convert("RGB")
        self.points = points
        self.heading.configure(text=f"{path.parent.name}/{path.name}  |  {self.image.width} × {self.image.height}")
        self.fit()

    def fit(self):
        if self.image is None:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.scale = min((width - 24) / self.image.width, (height - 24) / self.image.height)
        self.scale = max(self.scale, 0.001)
        self.offset_x = (width - self.image.width * self.scale) / 2
        self.offset_y = (height - self.image.height * self.scale) / 2
        self.draw()

    def resize(self, _event):
        self.fit()

    def to_image(self, x, y):
        # (0, 0) is the center of the first pixel, not its outer edge.
        return ((x - self.offset_x) / self.scale - 0.5,
                (y - self.offset_y) / self.scale - 0.5)

    def valid_point(self, x, y):
        return self.image is not None and 0 <= x <= self.image.width - 1 and 0 <= y <= self.image.height - 1

    def click(self, event):
        x, y = self.to_image(event.x, event.y)
        if self.valid_point(x, y):
            self.on_point([round(x, 3), round(y, 3)])

    def motion(self, event):
        x, y = self.to_image(event.x, event.y)
        if self.valid_point(x, y):
            self.coordinate_label.configure(text=f"原图坐标  x: {x:.1f}   y: {y:.1f}")

    def zoom(self, event):
        if self.image is None:
            return
        factor = 1.25 if event.delta > 0 else 0.8
        new_scale = min(12.0, max(0.03, self.scale * factor))
        factor = new_scale / self.scale
        self.offset_x = event.x - (event.x - self.offset_x) * factor
        self.offset_y = event.y - (event.y - self.offset_y) * factor
        self.scale = new_scale
        self.draw()

    def pan_start(self, event):
        self.drag = (event.x, event.y)

    def pan_move(self, event):
        if self.drag:
            self.offset_x += event.x - self.drag[0]
            self.offset_y += event.y - self.drag[1]
            self.drag = (event.x, event.y)
            self.draw()

    def draw(self):
        self.canvas.delete("all")
        if self.image is None:
            return
        # Render only the visible crop, so zooming never creates a huge bitmap.
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        left = max(0, int(math.floor(-self.offset_x / self.scale)))
        top = max(0, int(math.floor(-self.offset_y / self.scale)))
        right = min(self.image.width, int(math.ceil((width - self.offset_x) / self.scale)))
        bottom = min(self.image.height, int(math.ceil((height - self.offset_y) / self.scale)))
        if right > left and bottom > top:
            crop = self.image.crop((left, top, right, bottom))
            size = (max(1, round((right - left) * self.scale)),
                    max(1, round((bottom - top) * self.scale)))
            self.photo = ImageTk.PhotoImage(crop.resize(size, Image.Resampling.BILINEAR))
            self.canvas.create_image(self.offset_x + left * self.scale,
                                     self.offset_y + top * self.scale,
                                     anchor="nw", image=self.photo)
        for number, (x, y) in enumerate(self.points, 1):
            sx = self.offset_x + (x + 0.5) * self.scale
            sy = self.offset_y + (y + 0.5) * self.scale
            color = "#ffe04a"
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, outline="#111111", width=4)
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, outline=color, width=2)
            self.canvas.create_line(sx - 9, sy, sx + 9, sy, fill=color)
            self.canvas.create_line(sx, sy - 9, sx, sy + 9, fill=color)
            self.canvas.create_text(sx + 10, sy - 12, text=str(number), fill="#111111", anchor="sw", font=("Arial", 14, "bold"))
            self.canvas.create_text(sx + 9, sy - 13, text=str(number), fill=color, anchor="sw", font=("Arial", 14, "bold"))


class PointPicker:
    def __init__(self, root):
        self.root = root
        self.root.title("Assignment 3 — 手动对应点标注")
        self.root.geometry("1220x900")
        self.root.minsize(850, 600)
        self.paths = []
        for folder in sorted((ROOT / "photos").iterdir()):
            if folder.is_dir():
                images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
                self.paths.extend((a, b) for a, b in zip(images, images[1:]))
        if not self.paths:
            raise ValueError("photos 文件夹里没有相邻照片。")
        self.points = [[], []]
        self.index = 0
        self.save_error = False
        toolbar = ttk.Frame(root, padding=10)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="照片对：").pack(side="left")
        labels = [f"第 {a.parent.name} 组：{a.name} → {b.name}" for a, b in self.paths]
        self.selection = ttk.Combobox(toolbar, values=labels, state="readonly", width=27)
        default = next((i for i, (a, _) in enumerate(self.paths) if a.parent.name == "2"), 0)
        self.selection.current(default)
        self.selection.pack(side="left", padx=(0, 12))
        self.selection.bind("<<ComboboxSelected>>", self.change_pair)
        ttk.Button(toolbar, text="撤销上一次点击 (Ctrl+Z)", command=self.undo).pack(side="left", padx=4)
        ttk.Button(toolbar, text="恢复全图", command=self.fit).pack(side="left", padx=4)
        ttk.Button(toolbar, text="保存 (Ctrl+S)", command=self.save_clicked).pack(side="left", padx=4)
        self.status = ttk.Label(root, font=("Microsoft YaHei UI", 12, "bold"), anchor="center")
        self.status.pack(fill="x", padx=10, pady=(0, 5))
        ttk.Label(root, text="按顺序：左图点一个角 → 右图点同一个角。滚轮放大；按住右键拖动。建议 8–12 对，分散在重叠区域。", anchor="center").pack(fill="x")
        images_frame = ttk.Frame(root, padding=8)
        images_frame.pack(fill="both", expand=True)
        images_frame.columnconfigure((0, 1), weight=1, uniform="image")
        images_frame.rowconfigure(0, weight=1)
        self.panes = [ImagePane(images_frame, "左图", lambda p: self.add(0, p)),
                      ImagePane(images_frame, "右图", lambda p: self.add(1, p))]
        for i, pane in enumerate(self.panes):
            pane.grid(row=0, column=i, sticky="nsew", padx=4)
        self.table = ttk.Treeview(root, columns=("n", "x1", "y1", "x2", "y2"), show="headings", height=4)
        for column, title in zip(("n", "x1", "y1", "x2", "y2"), ("点对编号", "左图 x", "左图 y", "右图 x", "右图 y")):
            self.table.heading(column, text=title)
            self.table.column(column, anchor="center", width=110)
        self.table.pack(fill="x", padx=14, pady=4)
        self.footer = ttk.Label(root, anchor="center")
        self.footer.pack(fill="x", padx=10, pady=(2, 10))
        root.bind("<Control-z>", lambda _e: self.undo())
        root.bind("<Control-s>", lambda _e: self.save_clicked())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.load(default)

    def output_path(self):
        a, b = self.paths[self.index]
        return ROOT / "correspondences" / f"set_{a.parent.name}_{a.stem}_to_{b.stem}.json"

    def load(self, index):
        self.index = index
        self.points = [[], []]
        path = self.output_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                expected = [p.relative_to(ROOT).as_posix() for p in self.paths[index]]
                if [data["image1"], data["image2"]] != expected:
                    raise ValueError("保存文件中的照片路径不匹配")
                p1, p2 = data["im1_pts"], data["im2_pts"]
                if len(p1) != len(p2):
                    raise ValueError("保存文件中的点对数量不匹配")
                self.points = [p1, p2]
            for pane, image_path, points in zip(self.panes, self.paths[index], self.points):
                pane.load(image_path, points)
            for pane in self.panes:
                if any(len(p) != 2 or not all(math.isfinite(v) for v in p) or not pane.valid_point(*p) for p in pane.points):
                    raise ValueError("保存文件含无效或越界坐标")
        except (KeyError, TypeError, ValueError, OSError) as error:
            messagebox.showerror("无法读取点对", f"{path}\n{error}\n请先修复或移走这个文件后重新启动。")
            self.root.destroy()
            return
        self.save_error = False
        self.refresh()

    def change_pair(self, _event):
        target = self.selection.current()
        if len(self.points[0]) != len(self.points[1]):
            if not messagebox.askyesno("当前有一个未配对的点", "丢弃最后这个左图点并切换？已完成的点对已自动保存。"):
                self.selection.current(self.index)
                return
        if not self.save():
            self.selection.current(self.index)
            return
        self.load(target)

    def add(self, side, point):
        expected_side = 0 if len(self.points[0]) == len(self.points[1]) else 1
        if side != expected_side:
            self.root.bell()
            return
        self.points[side].append(point)
        if side == 1:
            self.save()
        self.refresh()

    def undo(self):
        side = 0 if len(self.points[0]) > len(self.points[1]) else 1
        if self.points[side]:
            self.points[side].pop()
            self.save()
            self.refresh()

    def fit(self):
        for pane in self.panes:
            pane.fit()

    def save(self):
        count = min(map(len, self.points))
        a, b = self.paths[self.index]
        data = {
            "image1": a.relative_to(ROOT).as_posix(),
            "image2": b.relative_to(ROOT).as_posix(),
            "image1_size": list(self.panes[0].image.size),
            "image2_size": list(self.panes[1].image.size),
            "coordinate_system": "EXIF-oriented original image; x right, y down; (0,0) first pixel center",
            "homography_direction": "image1 -> image2",
            "im1_pts": self.points[0][:count],
            "im2_pts": self.points[1][:count],
            "num_pairs": count,
        }
        path = self.output_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
            self.save_error = False
            return True
        except OSError as error:
            self.save_error = True
            messagebox.showerror("保存失败", str(error))
            return False

    def save_clicked(self):
        if self.save():
            self.refresh()
            suffix = "\n最后一个左图点尚未配对，未写入文件。" if len(self.points[0]) != len(self.points[1]) else ""
            messagebox.showinfo("保存成功", f"已保存 {min(map(len, self.points))} 对坐标：\n{self.output_path()}{suffix}")

    def refresh(self):
        count = len(self.points[1])
        pending = len(self.points[0]) > count
        next_step = f"请在右图点击第 {count + 1} 个点的相同位置" if pending else f"请在左图点击第 {count + 1} 个点"
        self.status.configure(text=f"已完成 {count} 对  |  {next_step}")
        for i, pane in enumerate(self.panes):
            pane.canvas.configure(highlightbackground="#35bd90" if i == int(pending) else "#6a727e")
            pane.draw()
        self.table.delete(*self.table.get_children())
        for i, left in enumerate(self.points[0]):
            right = self.points[1][i] if i < count else ("待点击", "待点击")
            item = self.table.insert("", "end", values=(i + 1, *left, *right))
            self.table.see(item)
        saved = "保存失败，请重试" if self.save_error else ("完成的点对已自动保存" if self.output_path().exists() else "每完成一对将自动保存")
        self.footer.configure(text=f"{saved}：{self.output_path().relative_to(ROOT)}")

    def close(self):
        if len(self.points[0]) != len(self.points[1]):
            if not messagebox.askyesno("有一个未配对的点", "丢弃最后这个左图点并退出？已完成的点对会保存。"):
                return
        if self.save():
            self.root.destroy()


def main():
    root = tk.Tk()
    ttk.Style().configure(".", font=("Microsoft YaHei UI", 10))
    PointPicker(root)
    root.mainloop()


if __name__ == "__main__":
    main()
