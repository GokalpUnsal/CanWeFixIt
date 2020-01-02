import cv2
import numpy as np
import wx
from wx.core import wx


class CanWeFixItGUI(wx.Frame):
    MAX_WIDTH = 512
    MAX_HEIGHT = 512

    def __init__(self, parent, id, title):
        wx.Frame.__init__(self,
                          parent,
                          id,
                          title,
                          wx.DefaultPosition,
                          wx.Size(800, 800))

        self.dirname = ''
        self.filename = ''
        self.img = ''
        self.original_img = None
        self.painted_img = None
        self.mask_img = None
        self.original_img_width = 0
        self.original_img_height = 0
        self.img_width = 0
        self.img_height = 0
        self.drawing = False
        self.prev_point = None
        self.wx_bitmap = None

        self.CreateStatusBar()

        wx.Button(self, 1, 'Inpaint', (10, 130))
        wx.Button(self, 2, 'Load Image', (10, 10))
        wx.Button(self, 3, 'Export Mask', (10, 50))
        wx.Button(self, 4, 'Generate Random Mask', (10, 90))
        self.Bind(wx.EVT_BUTTON, self.onInpaint, id=1)
        self.Bind(wx.EVT_BUTTON, self.onOpenImage, id=2)
        self.Bind(wx.EVT_BUTTON, self.onExportMask, id=3)
        self.Bind(wx.EVT_BUTTON, self.onGenerateRandomMask, id=4)

        self.Centre()
        self.Show()

    def onOpenImage(self, e):
        self.reset_state()

        wildcard = 'JPEG files (*.jpg)|*.jpg|' + \
                   'PNG files (*.png)|*.png|' + \
                   'Other files (*.*)|*.*'
        dlg = wx.FileDialog(self, "Choose an image", wildcard=wildcard, style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            self.img = self.dirname + '/' + self.filename

        original_img = cv2.imread(self.img, cv2.IMREAD_COLOR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        self.original_img_height = original_img.shape[0]
        self.original_img_width = original_img.shape[1]

        if self.original_img_height > self.original_img_width:
            self.img_height = self.MAX_HEIGHT
            self.img_width = int(self.original_img_width * self.MAX_HEIGHT / self.original_img_height)
        else:
            self.img_height = int(self.original_img_height * self.MAX_WIDTH / self.original_img_width)
            self.img_width = self.MAX_WIDTH

        self.original_img = cv2.resize(original_img, (self.img_width, self.img_height))
        self.painted_img = cv2.resize(original_img, (self.img_width, self.img_height))
        self.mask_img = np.zeros((self.img_height, self.img_width, 1), np.uint8)

        wx_img = wx.Bitmap.FromBuffer(self.img_width, self.img_height, self.painted_img)
        self.wx_bitmap = wx.StaticBitmap(self, -1, wx_img, (106, 25))
        self.wx_bitmap.Bind(wx.EVT_LEFT_DOWN, self.on_image_left_down)
        self.wx_bitmap.Bind(wx.EVT_LEFT_UP, self.on_image_left_up)
        self.wx_bitmap.Bind(wx.EVT_MOTION, self.on_image_motion)

        dlg.Destroy()

    def onInpaint(self, e):
        pass

    def onExportMask(self, e):
        cv2.imwrite("mask_%s" % self.filename, self.mask_img)

    def onGenerateRandomMask(self, e):
        """Generate mask tensor from bbox.
            Returns:
                tf.Tensor: output with shape [1, H, W, 1]
            """
        min_num_vertex = 4
        max_num_vertex = 12
        mean_angle = 2 * np.math.pi / 5
        angle_range = 2 * np.math.pi / 15
        min_width = 12
        max_width = 40
        H = 256
        W = 256

        average_radius = np.math.sqrt(H * H + W * W) / 8
        mask = np.zeros((H, W, 1), np.uint8)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * np.math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.shape[:2]
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            width = int(np.random.uniform(min_width, max_width))
            for i in range(1, num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * np.math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * np.math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))
                cv2.line(mask, vertex[i - 1], vertex[i], (255, 255, 255), thickness=width)
                cv2.circle(mask,  vertex[i], width // 2, (255, 255, 255), thickness=-1)
        if np.random.normal() > 0:
            cv2.flip(mask, 0)
        if np.random.normal() > 0:
            cv2.flip(mask, 1)

        cv2.imwrite("random_mask.png", mask)

    def on_image_left_down(self, e):
        x, y = e.GetPosition()
        self.drawing = True
        self.prev_point = None

        self.draw_image(x, y)

    def on_image_left_up(self, e):
        self.drawing = False
        self.prev_point = None

    def on_image_motion(self, e):
        if self.drawing:
            x, y = e.GetPosition()
            self.draw_image(x, y)

    def draw_image(self, x, y):
        if self.prev_point is None:
            cv2.circle(self.painted_img, (x, y), 10, (255, 255, 255), -1)
            cv2.circle(self.mask_img, (x, y), 10, (255, 255, 255), -1)
        else:
            cv2.line(self.painted_img, self.prev_point, (x, y), (255, 255, 255), thickness=20)
            cv2.line(self.mask_img, self.prev_point, (x, y), (255, 255, 255), thickness=20)

        self.wx_bitmap.SetBitmap(wx.Bitmap.FromBuffer(self.img_width, self.img_height, self.painted_img))
        self.wx_bitmap.Refresh()
        self.prev_point = (x, y)

    def reset_state(self):
        if self.wx_bitmap:
            self.wx_bitmap.Destroy()
        self.dirname = ''
        self.filename = ''
        self.img = ''
        self.original_img = None
        self.painted_img = None
        self.mask_img = None
        self.original_img_width = 0
        self.original_img_height = 0
        self.img_width = 0
        self.img_height = 0
        self.drawing = False
        self.prev_point = None
        self.wx_bitmap = None


if __name__ == '__main__':
    app = wx.App(False)
    frame = CanWeFixItGUI(None, -1, "Can We Fix It")
    app.MainLoop()
